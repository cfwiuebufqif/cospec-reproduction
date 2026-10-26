"""
llm_backends.py — 三个 LLM 后端：vllm / openai / mock。

- vllm: 本机（RTX 4090）vLLM 服务暴露的 OpenAI 兼容端点；
- openai: 任意 OpenAI 兼容 chat API（官方 OpenAI、DeepSeek、Qwen 兼容端点等）；
- mock: 离线确定性占位后端，供本机端到端自测；其全部数字均为合成占位，
        输出 JSON 内带 [MOCK-PLACEHOLDER] 标记，结果文件 provenance 字段为
        "selftest-mock-placeholder"，禁止写入论文。

仅使用标准库 urllib（不强制依赖 requests）；temperature 默认 0.3，与原实验
cost_probe.TEMPERATURE / anchoring_probe_v4.TEMPERATURE 对齐。

环境变量
--------
vllm:
    COSPEC_VLLM_URL    默认 http://localhost:8000/v1/chat/completions
    COSPEC_VLLM_MODEL  模型名（也可由 CLI --model 传入；二者缺一即报错）
    COSPEC_VLLM_API_KEY 可选；设置后附加 Authorization: Bearer <key>
openai:
    OPENAI_BASE_URL    必填，例如 https://api.deepseek.com/v1
    OPENAI_API_KEY     必填；兼容原 cost_probe.py 的 DEEPSEEK_API_KEY 回退
    （二者均缺失时在发起请求前抛出清晰报错，不会静默调用）
mock:
    无环境变量；不联网、不睡眠、结果确定性可复现。
"""
import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

TEMPERATURE = 0.3  # 与原实验一致（cost_probe.py / anchoring_probe_v4.py）


def _post_json(url, payload, headers, timeout):
    """标准库 POST JSON，返回解析后的 dict。"""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} 调用失败: {body[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"网络连接失败 ({url}): {e.reason}") from e


class BaseBackend:
    """后端公共接口：call(prompt) -> 原始文本（与原 call_model 返回值一致）。"""

    backend_name = "base"

    def __init__(self, model, temperature=TEMPERATURE, timeout=120, sleep=0.2):
        self.model = model
        self.temperature = float(temperature)
        self.timeout = timeout
        self.sleep = sleep

    @property
    def provenance(self):
        return f"real-data-{self.backend_name}:{self.model}"

    def call(self, prompt):
        raise NotImplementedError

    def _chat_payload(self, prompt):
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

    @staticmethod
    def _extract_content(data):
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                "响应格式不符合 OpenAI 兼容 schema（缺少 choices[0].message.content）"
            ) from e


class VLLMBackend(BaseBackend):
    """本机 vLLM 服务（OpenAI 兼容 /chat/completions）。"""

    backend_name = "vllm"

    def __init__(self, model=None, temperature=TEMPERATURE, timeout=120, sleep=0.0):
        self.url = os.environ.get(
            "COSPEC_VLLM_URL", "http://localhost:8000/v1/chat/completions")
        model = model or os.environ.get("COSPEC_VLLM_MODEL")
        if not model:
            raise RuntimeError(
                "vllm 后端缺少模型名：请传 --model，或设置环境变量 COSPEC_VLLM_MODEL。")
        super().__init__(model, temperature, timeout, sleep)

    def call(self, prompt):
        headers = {"Content-Type": "application/json"}
        key = os.environ.get("COSPEC_VLLM_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        data = _post_json(self.url, self._chat_payload(prompt), headers,
                          self.timeout)
        out = self._extract_content(data)
        if self.sleep:
            time.sleep(self.sleep)
        return out


class OpenAIBackend(BaseBackend):
    """OpenAI 兼容远程 API；缺 base_url / key 时拒绝发起请求。

    正式采集走 ``prefill_all`` 并发预取：远程 API 不吃本地算力，逐条往返会
    被网络延迟主导（推理模型单次可达数十秒），故用线程池并发，线程数由环境
    变量 ``COSPEC_API_CONCURRENCY`` 控制（默认 8）。并发只改变请求的到达
    顺序，每次调用彼此独立、参数与温度完全一致，结果按 prompts 顺序回灌，
    不影响可复现性。

    断点续跑（省钱，重要）：响应按 sha256(model|temperature|prompt) 增量落盘
    到 ``COSPEC_RESPONSE_CACHE`` 指定的 JSONL（每条完成即追加并 fsync）。
    重跑时先读缓存，已取到的 prompt 直接复用、不再发请求扣费；因此中途余额
    不足（HTTP 402）或限流中断后，充值后重跑同一命令即可只补缺失部分。
    429/5xx/超时指数退避重试；402 余额不足立即停止派发、不再无意义重试。
    """

    backend_name = "openai"

    def __init__(self, model, temperature=TEMPERATURE, timeout=120, sleep=0.0,
                 concurrency=None, max_retries=8, cache_path=None):
        base = os.environ.get("OPENAI_BASE_URL")
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
            "DEEPSEEK_API_KEY")
        if not base:
            raise RuntimeError(
                "openai 后端缺少端点：请设置环境变量 OPENAI_BASE_URL"
                "（例如 https://api.deepseek.com/v1）。")
        if not key:
            raise RuntimeError(
                "openai 后端缺少密钥：请设置环境变量 OPENAI_API_KEY"
                "（兼容原 cost_probe.py 的 DEEPSEEK_API_KEY 回退）。")
        self.url = base.rstrip("/") + "/chat/completions"
        self.key = key
        self.concurrency = int(concurrency or os.environ.get(
            "COSPEC_API_CONCURRENCY", "8"))
        self.max_retries = int(max_retries)
        self.cache_path = cache_path or os.environ.get("COSPEC_RESPONSE_CACHE")
        self._cache = {}
        self._disk = {}
        self._file_lock = threading.Lock()
        self._load_disk()
        super().__init__(model, temperature, timeout, sleep)

    def _cache_key(self, prompt):
        raw = f"{self.model}|{self.temperature}|{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load_disk(self):
        if not self.cache_path:
            return
        path = Path(self.cache_path)
        if not path.exists():
            return
        n = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._disk[rec["k"]] = rec["v"]
                    n += 1
                except (ValueError, KeyError):
                    continue
        print(f"[openai] 加载响应缓存 {n} 条: {self.cache_path}", flush=True)

    def _append_disk(self, key, value):
        if not self.cache_path:
            return
        path = Path(self.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"k": key, "v": value},
                                   ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

    @staticmethod
    def _is_balance_error(msg):
        return ("HTTP 402" in msg or "Insufficient Balance" in msg
                or "insufficient_balance" in msg)

    def _one(self, prompt):
        """单次 chat 调用；429/5xx/超时指数退避（带抖动），402 立即抛。"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.key}",
        }
        last = None
        for attempt in range(self.max_retries):
            try:
                data = _post_json(self.url, self._chat_payload(prompt),
                                  headers, self.timeout)
                return self._extract_content(data)
            except RuntimeError as e:
                last = e
                msg = str(e)
                if self._is_balance_error(msg):
                    raise
                retryable = ("HTTP 429" in msg or "HTTP 409" in msg
                             or "HTTP 500" in msg or "HTTP 502" in msg
                             or "HTTP 503" in msg or "HTTP 504" in msg
                             or "timed out" in msg.lower()
                             or "网络连接失败" in msg)
                if retryable:
                    wait = min(1.0 * (2 ** attempt), 45.0)
                    time.sleep(wait + random.uniform(0, 1.5))
                    continue
                raise
        raise last

    def prefill_all(self, prompts):
        """去重后并发预取；命中磁盘缓存的跳过、新结果增量落盘，断点可续。"""
        unique = list(dict.fromkeys(prompts))
        results = {}
        pending = []
        for p in unique:
            key = self._cache_key(p)
            if key in self._disk:
                results[p] = self._disk[key]
            else:
                pending.append(p)
        total = len(unique)
        hit = total - len(pending)
        print(f"[openai] 缓存命中 {hit}/{total}，待请求 {len(pending)}",
              flush=True)

        balance_failed = 0
        if pending:
            done = hit
            with ThreadPoolExecutor(max_workers=max(1, self.concurrency)) as pool:
                futs = {pool.submit(self._one, p): p for p in pending}
                for fut in as_completed(list(futs)):
                    p = futs[fut]
                    try:
                        value = fut.result()
                    except Exception as e:
                        if self._is_balance_error(str(e)):
                            balance_failed += 1
                            # 余额不足：取消尚未开始的请求，避免继续无效扣费
                            for other in futs:
                                if not other.done():
                                    other.cancel()
                            break
                        # 其他瞬时错误（非余额）：本条留空，稍后整体报缺失
                        continue
                    key = self._cache_key(p)
                    self._disk[key] = value
                    self._append_disk(key, value)
                    results[p] = value
                    done += 1
                    new_count = done - hit
                    if new_count % 25 == 0 or done == total:
                        print(f"  [openai] prefill {done}/{total}", flush=True)

        missing = [p for p in unique if p not in results]
        if missing:
            reason = "余额不足(HTTP 402)，请充值后重跑" if balance_failed else "限流/错误"
            hint = (f"；已完成 {len(results)}/{total} 条已落盘 {self.cache_path}，"
                    f"重跑同一命令只补缺失、不重复扣费"
                    if self.cache_path else "")
            raise RuntimeError(
                f"[openai] {len(missing)} 条未完成（{reason}）{hint}")

        bucket = {}
        for p in prompts:
            bucket.setdefault(p, []).append(results[p])
        self._cache = {p: iter(v) for p, v in bucket.items()}
        return len(prompts)

    def call(self, prompt):
        seq = self._cache.get(prompt)
        if seq is not None:
            nxt = next(seq, None)
            if nxt is not None:
                return nxt
        out = self._one(prompt)
        if self.sleep:
            time.sleep(self.sleep)
        return out


class MockBackend(BaseBackend):
    """离线确定性占位后端。

    仅用于本机端到端自测：根据 prompt 文本的 md5 派生固定种子，生成落在原
    prompt 区间内的占位数值；原始回复内带 [MOCK-PLACEHOLDER] 标记。同一条
    prompt 在任何机器、任何运行次数下结果一致。其数字不代表任何真实模型行为。
    """

    backend_name = "mock"

    def __init__(self, model="mock-placeholder", temperature=TEMPERATURE,
                 timeout=1, sleep=0.0):
        super().__init__(model or "mock-placeholder", temperature, timeout, sleep)

    @property
    def provenance(self):
        return "selftest-mock-placeholder"

    @staticmethod
    def _seed(prompt):
        digest = hashlib.md5(prompt.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little")

    def call(self, prompt):
        rng = np.random.default_rng(self._seed(prompt))
        placeholder = ("[MOCK-PLACEHOLDER] deterministic offline self-test "
                       "answer; NOT a real LLM response, must not enter any "
                       "reported result.")
        if "omega_n" in prompt:
            # 与 anchoring_probe_v4.PROMPT 的 mid 区间对齐：
            # omega_n [0.1, 2.0]，zeta [0.05, 2.0]。
            wn = round(float(rng.uniform(0.1, 2.0)), 4)
            z = round(float(rng.uniform(0.05, 2.0)), 4)
            payload = {
                "omega_n": wn,
                "zeta": z,
                "reasoning": placeholder,
            }
        else:
            q1 = round(float(rng.uniform(2.0, 20.0)), 3)
            q2 = round(float(rng.uniform(0.2, 3.0)), 3)
            rr = round(float(rng.uniform(0.1, 2.0)), 3)
            payload = {
                "q1": q1,
                "q2": q2,
                "r": rr,
                "reasoning": placeholder,
            }
        return f"{placeholder}\n{json.dumps(payload)}"


class HFBackend(BaseBackend):
    """本机 transformers 加载的开源模型（bf16、GPU），支持批量预生成。

    用于不便安装 vLLM（例如最新 Blackwell 架构搭配较新 torch，vLLM 会要求
    降级 torch）时的本地推理；只依赖 transformers，不动 torch 版本。
    ``model`` 为 HF 模型 ID 或本地权重目录。正式采集前应先调用
    ``prefill_all(prompts)``，一次性按批生成全部 prompt（远快于逐条）；
    之后 ``call()`` 顺序返回缓存结果。未预生成的 prompt 会单条即时生成。
    """

    backend_name = "hf"

    def __init__(self, model, temperature=TEMPERATURE, timeout=120, sleep=0.0,
                 batch_size=32, max_input_tokens=3072, max_new_tokens=256):
        if not model:
            raise RuntimeError(
                "hf 后端必须通过 --model 指定 HF 模型 ID 或本地权重目录。")
        super().__init__(model, temperature, timeout, sleep)
        self.batch_size = int(batch_size)
        self.max_input_tokens = int(max_input_tokens)
        self.max_new_tokens = int(max_new_tokens)
        self._torch = None
        self._model = None
        self._tok = None
        self._cache = {}
        self._load()

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(
            self.model, padding_side="left", trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model, dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True)
        self._model.eval()
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token

    def _generate_batch(self, prompts):
        torch = self._torch
        messages = [[{"role": "user", "content": p}] for p in prompts]
        texts = [self._tok.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True) for m in messages]
        enc = self._tok(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=self.max_input_tokens)
        enc = {k: v.to(self._model.device) for k, v in enc.items()}
        gen_kwargs = dict(max_new_tokens=self.max_new_tokens,
                          do_sample=self.temperature > 0,
                          pad_token_id=self._tok.pad_token_id)
        if self.temperature > 0:
            gen_kwargs.update(temperature=float(self.temperature), top_p=1.0)
        with torch.no_grad():
            out = self._model.generate(**enc, **gen_kwargs)
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        return self._tok.batch_decode(new_tokens, skip_special_tokens=True)

    def prefill_all(self, prompts):
        """按出现顺序批量生成；同一 prompt 重复（repeats）的结果按序缓存。"""
        from collections import defaultdict
        cache = defaultdict(list)
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            outs = self._generate_batch(batch)
            for p, o in zip(batch, outs):
                cache[p].append(o)
        self._cache = {p: iter(v) for p, v in cache.items()}
        return len(prompts)

    def call(self, prompt):
        seq = self._cache.get(prompt)
        if seq is not None:
            nxt = next(seq, None)
            if nxt is not None:
                return nxt
        return self._generate_batch([prompt])[0]


def make_backend(name, model=None, temperature=TEMPERATURE, timeout=120,
                 sleep=0.2):
    """后端工厂；未知名称抛出 ValueError 并列出可选项。"""
    name = (name or "").strip().lower()
    if name == "vllm":
        return VLLMBackend(model=model, temperature=temperature,
                           timeout=timeout, sleep=sleep)
    if name == "openai":
        if not model:
            raise RuntimeError("openai 后端必须通过 --model 指定模型名。")
        return OpenAIBackend(model=model, temperature=temperature,
                              timeout=timeout, sleep=sleep)
    if name == "hf":
        return HFBackend(model=model, temperature=temperature,
                         timeout=timeout, sleep=sleep)
    if name == "mock":
        return MockBackend(model=model, temperature=temperature)
    raise ValueError(
        f"未知后端 '{name}'；可选: vllm | openai | hf | mock")
