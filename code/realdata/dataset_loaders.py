"""
dataset_loaders.py — 真实心理文本数据集统一加载层（CoSpec 真实数据适配层）。

统一记录 schema:
    {"id": str, "text": str, "meta": dict}
  - text: 求助者第一人称心理文本，直接替换原实验中的 vignette 文本喂给原 prompt；
  - meta: 来源溯源信息（数据集名、源行 id、话题、许可、保留意见等），不进入 prompt。

所有加载器只读本地文件，不发起任何网络请求；下载命令见 README.md。
路径一律用 os.path 拼接，跨平台（Windows / Ubuntu）通用。

选型口径对齐同级交付物《数据集选型报告》§9（最终选型由项目负责人拍板）:
  9.1 主推荐  CounselChat        HF nbertagnolli/counsel-chat，MIT，940 个真实问题
  9.2 备选一  CounselBench-Eval  HF izi-ano/CounselBench-Eval，CC BY-NC-ND 4.0，
                                100 个真实问题复测集；许可受限，须合规复核
  9.3 备选二  MentalChat16K       仅 Interview_Data_6K.csv（家属转述、模型改写的
                                二手长叙事，须单独标注）；Synthetic_Data_10K.csv
                                为 GPT-3.5 合成，加载器拒绝读取、不计入真实输入

字段核实状态（2026-10-23 联网核实，与选型报告 §5.4 / §5.7 / §4.4 一致；未能在
本机打开原文核对的条目在注册表 pending_verification 中标注"待核实"，不做推断填补）。
"""
import csv
import json
import os
import statistics

_HERE = os.path.dirname(os.path.abspath(__file__))
_SELFTEST_PATH = os.path.join(_HERE, "data", "selftest_data.jsonl")

# 数据集注册表：README.md 与 CLI --dataset 选项共用，记录来源、许可、核实状态与
# 必须从输入上下文剥离的咨询师侧 / 回答侧 / 评分侧列。
DATASET_REGISTRY = {
    "counselchat": {
        "role": "primary",
        "kind": "csv",
        "source": "https://huggingface.co/datasets/nbertagnolli/counsel-chat",
        "github": "https://github.com/nbertagnolli/counsel-chat",
        "license": "MIT",
        "verified_columns": [
            "questionID", "questionTitle", "questionText", "questionLink",
            "topic", "therapistInfo", "therapistURL", "answerText",
            "upvotes", "views",
        ],
        "text_columns": ["questionText", "questionTitle"],
        "id_column": "questionID",
        # 咨询师侧信息与互动计数：只入溯源 meta 也不保留，一律剥离，禁止进入 LLM 上下文。
        "strip_columns": [
            "answerText", "therapistInfo", "therapistURL", "upvotes", "views",
        ],
        "default_local_file": os.path.join("data", "counselchat",
                                           "20220401_counsel_chat.csv"),
        "expected": {"raw_rows": 2775, "unique_questions": 940, "topics": 31},
        "pending_verification": [
            "GitHub data/ 三张旧快照（20200325_counsel_chat.csv、counsel_chat_250-tokens_full.json、"
            "counselchat-data.csv）与 HF 20220401 版的字段差异未逐张比对，字段以 HF 主表为准",
            "非官方镜像 mpingale/mental-health-chat-dataset 的可用性与列名待核实，不作为默认来源",
        ],
        "note": "2022-04-01 爬取自 counselchat.com 的匿名求助者英文第一人称求助帖；"
                "同一问题可对应多条咨询师回答，按 questionID 去重为一条来访者记录。",
    },
    "counselbench_eval": {
        "role": "backup_eval",
        "kind": "csv",
        "source": "https://huggingface.co/datasets/izi-ano/CounselBench-Eval",
        "license": "CC BY-NC-ND 4.0（非商业使用、禁止衍生作品；仅限内部研究使用，"
                   "对外发布或二次分发前必须完成法务/合规复核）",
        "verified_columns": [
            "questionID", "questionTitle", "questionText", "response", "topic",
            "responder", "survey_id", "overall_score", "overall_reason",
            "empathy_score", "specificity_score", "medical_advice_score",
            "medical_copy", "medical_reason", "factual_consistency_score",
            "factual_copy", "factual_reason", "toxicity_score", "toxicity_copy",
            "toxicity_reason",
        ],
        "text_columns": ["questionText", "questionTitle"],
        "id_column": "questionID",
        # 回答、回答方、问卷 id 与全部六维评分/理由/标注列：本实验一律剥离，防止
        # 结构化量表信息泄露进冻结 LLM 的上下文（选型报告 §5.9 风险点 2）。
        "strip_columns": [
            "response", "responder", "survey_id",
            "overall_score", "overall_reason",
            "empathy_score", "specificity_score", "medical_advice_score",
            "medical_copy", "medical_reason",
            "factual_consistency_score", "factual_copy", "factual_reason",
            "toxicity_score", "toxicity_copy", "toxicity_reason",
        ],
        "default_local_file": os.path.join("data", "counselbench",
                                           "counselbench_eval.csv"),
        "expected": {"raw_rows": 2000, "unique_questions": 100},
        "pending_verification": [
            "CC BY-NC-ND 4.0 条款在本项目内的法务复核结论待项目合规口径确认（选型报告 §10.1 第 11 条）",
        ],
        "note": "100 个经临床筛选的真实 CounselChat 问题（20 话题 × 5 个高赞问题），"
                "仅作小规模复测/一致性集；问题原文上游为 MIT CounselChat，但打包标注集按 "
                "NC-ND 发布，不能据此主张整包 MIT。",
    },
    "mentalchat16k": {
        "role": "backup_narrative",
        "kind": "csv",
        "source": "https://huggingface.co/datasets/ShenLab/MentalChat16K",
        "github": "https://github.com/PennShenLab/MentalChat16K",
        "license": "MIT",
        "verified_columns": ["instruction", "input", "output"],
        "text_columns": ["input"],
        "id_column": None,
        "strip_columns": ["instruction", "output"],
        "default_local_file": os.path.join("data", "mentalchat16k",
                                           "Interview_Data_6K.csv"),
        # 合成子集明确排除：禁止进入真实输入口径（选型报告 §4.6 风险点 2、§9.3）。
        "excluded_files": ["Synthetic_Data_10K.csv"],
        "pending_verification": [
            "Synthetic_Data_10K.csv 磁盘表头逐列复核（该文件已被排除，不进入真实输入口径）",
        ],
        "note": "仅 Interview_Data_6K.csv 可作补充长叙事备选：6,338 条源自 378 段行为健康"
                "教练—家属访谈转写，经 Mistral-7B-Instruct-v0.2 逐页改写/摘要；说话人是"
                "家属（caregiver）而非患者本人，属二手转述稿，报告时须与患者第一人称文本"
                "区分标注。Synthetic_Data_10K.csv 为 GPT-3.5 Turbo 合成，加载器拒绝读取。",
    },
    "jsonl": {
        "role": "generic",
        "kind": "jsonl",
        "source": "用户自备 JSONL，每行一个记录",
        "license": "随用户自备数据，许可由用户自行负责",
        "verified_columns": [],
        "text_columns": ["text"],
        "id_column": "id",
        "strip_columns": [],
        "pending_verification": [],
        "default_local_file": None,
        "note": "兜底通用格式；字段名由命令行 --text-field / --id-field 配置。",
    },
    "selftest": {
        "role": "selftest",
        "kind": "jsonl",
        "source": "随包合成自测占位数据（非真实临床文本，禁止写入论文）",
        "license": "占位自测",
        "verified_columns": ["id", "text", "meta"],
        "text_columns": ["text"],
        "id_column": "id",
        "strip_columns": [],
        "pending_verification": [],
        "default_local_file": _SELFTEST_PATH,
        "note": "6 条合成占位文本，仅用于离线端到端流程验证。",
    },
}


# --------------------------------------------------------------------- helpers
def _clean(text):
    """折叠空白并去首尾；None/空串归一为 None。"""
    if text is None:
        return None
    text = " ".join(str(text).split())
    return text if text else None


def _read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
    return rows, reader.fieldnames or []


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln} JSON 解析失败: {e}")
    return rows


def _pick(rec, candidates):
    """按候选列名顺序取第一个非空值。"""
    for col in candidates:
        if col in rec:
            val = _clean(rec[col])
            if val is not None:
                return val, col
    return None, None


def _require_file(name, path):
    if path is None:
        path = DATASET_REGISTRY[name]["default_local_file"]
    if path is None:
        raise ValueError(
            "--dataset jsonl 必须同时用 --dataset-path 指定本地 JSONL 文件")
    if not os.path.isabs(path):
        path = os.path.join(_HERE, path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"数据集文件不存在: {path}\n下载命令见 README.md 第 3.2 节。")
    return path


def _missing_columns(rec, required):
    return [c for c in required if c not in rec]


# ------------------------------------------------------------- per-dataset loaders
def load_counselchat(path=None, n=None):
    """CounselChat（主推荐，选型报告 §9.1）：questionText 回退 questionTitle，
    按 questionID 去重；咨询师回答、咨询师信息、点赞/浏览数全部剥离，不进 meta。"""
    path = _require_file("counselchat", path)
    rows, fields = _read_csv(path)
    reg = DATASET_REGISTRY["counselchat"]
    missing = _missing_columns(rows[0], reg["verified_columns"]) if rows else fields
    if missing:
        raise ValueError(
            f"{path} 缺少已核实列 {missing}；现有列: {fields}（字段核实状态见 README 3.3）")
    seen = {}
    for i, rec in enumerate(rows):
        qid = _clean(rec.get("questionID")) or f"row-{i}"
        if qid in seen:
            continue
        text, used = _pick(rec, reg["text_columns"])
        if text is None:
            continue
        seen[qid] = {
            "id": f"cc-{qid}",
            "text": text,
            "meta": {
                "dataset": "counselchat",
                "role": reg["role"],
                "question_id": qid,
                "topic": _clean(rec.get("topic")),
                "question_title": _clean(rec.get("questionTitle")),
                "question_link": _clean(rec.get("questionLink")),
                "source_row": i,
                "text_column": used,
                "license": reg["license"],
                "source": reg["source"],
                "speaker": "anonymous help-seeker (first-person forum post)",
                "verbatim": True,
                "stripped_columns": reg["strip_columns"],
            },
        }
    out = list(seen.values())
    if not out:
        raise ValueError(
            f"{path} 中未匹配到已知文本列 {reg['text_columns']}；现有列: {fields}")
    return out[:n] if n else out


def load_counselbench_eval(path=None, n=None):
    """CounselBench-Eval（备选复测集，选型报告 §9.2）：仅取 questionText/questionTitle，
    按 questionID 去重为 100 个问题；response 与全部评分/标注列剥离；meta 显式标注
    NC-ND 许可限制。"""
    path = _require_file("counselbench_eval", path)
    rows, fields = _read_csv(path)
    reg = DATASET_REGISTRY["counselbench_eval"]
    missing = _missing_columns(rows[0], reg["verified_columns"]) if rows else fields
    if missing:
        raise ValueError(
            f"{path} 缺少已核实列 {missing}；现有列: {fields}（字段核实状态见 README 3.3）")
    seen = {}
    for i, rec in enumerate(rows):
        qid = _clean(rec.get("questionID")) or f"row-{i}"
        if qid in seen:
            continue
        text, used = _pick(rec, reg["text_columns"])
        if text is None:
            continue
        seen[qid] = {
            "id": f"cb-{qid}",
            "text": text,
            "meta": {
                "dataset": "counselbench_eval",
                "role": reg["role"],
                "question_id": qid,
                "topic": _clean(rec.get("topic")),
                "question_title": _clean(rec.get("questionTitle")),
                "source_row": i,
                "text_column": used,
                "license": reg["license"],
                "license_restricted": True,
                "source": reg["source"],
                "speaker": "anonymous help-seeker (question reused from CounselChat)",
                "verbatim": True,
                "stripped_columns": reg["strip_columns"],
                "usage_note": "仅限内部研究；对外发布或二次分发前须完成法务复核。",
            },
        }
    out = list(seen.values())
    if not out:
        raise ValueError(
            f"{path} 中未匹配到已知文本列 {reg['text_columns']}；现有列: {fields}")
    return out[:n] if n else out


def load_mentalchat16k(path=None, n=None):
    """MentalChat16K（备选长叙事，选型报告 §9.3）：仅允许 Interview_Data_6K.csv；
    Synthetic_Data_10K.csv 为 GPT-3.5 合成，直接拒绝读取。meta 显式标注二手转述性质。"""
    path = _require_file("mentalchat16k", path)
    base = os.path.basename(path).lower()
    for excluded in DATASET_REGISTRY["mentalchat16k"]["excluded_files"]:
        if excluded.lower() in base:
            raise ValueError(
                f"{excluded} 为 GPT-3.5 Turbo 合成对话，按选型报告 §4.6 / §9.3 不纳入"
                "真实输入口径；本加载器拒绝读取该文件。")
    rows, fields = _read_csv(path)
    reg = DATASET_REGISTRY["mentalchat16k"]
    missing = _missing_columns(rows[0], reg["verified_columns"]) if rows else fields
    if missing:
        raise ValueError(
            f"{path} 缺少已核实列 {missing}；现有列: {fields}（字段核实状态见 README 3.3）")
    out = []
    for i, rec in enumerate(rows):
        text, used = _pick(rec, reg["text_columns"])
        if text is None:
            continue
        out.append({
            "id": f"mc16k-{i}",
            "text": text,
            "meta": {
                "dataset": "mentalchat16k",
                "role": reg["role"],
                "source_row": i,
                "text_column": used,
                "source_file": os.path.basename(path),
                "license": reg["license"],
                "source": reg["source"],
                "speaker": "caregiver (family member of patient), NOT the patient",
                "verbatim": False,
                "rewritten_by": "Mistral-7B-Instruct-v0.2",
                "secondary_hand_note": "家属访谈经模型改写/摘要的二手转述稿，报告时须与"
                                       "患者本人第一人称文本区分，不得合并为同一口径。",
                "stripped_columns": reg["strip_columns"],
            },
        })
    if not out:
        raise ValueError(
            f"{path} 中未匹配到已知文本列 {reg['text_columns']}；现有列: {fields}")
    return out[:n] if n else out


def load_jsonl(path, text_field="text", id_field="id", n=None):
    """通用 JSONL 兜底；除 id/text 外的全部字段进入 meta。"""
    path = _require_file("jsonl", path)
    rows = _read_jsonl(path)
    out = []
    for i, rec in enumerate(rows):
        text = _clean(rec.get(text_field))
        if text is None:
            continue
        rid = _clean(rec.get(id_field)) or f"jsonl-{i}"
        meta = {k: v for k, v in rec.items()
                if k not in (text_field, id_field)}
        meta.setdefault("dataset", "jsonl")
        meta["source_row"] = i
        out.append({"id": str(rid), "text": text, "meta": meta})
    if not out:
        raise ValueError(f"{path} 中未在列 '{text_field}' 读到非空文本。")
    return out[:n] if n else out


def load_selftest(path=None, n=None):
    """随包自测占位数据；meta.provenance 标记为合成占位。"""
    path = path or _SELFTEST_PATH
    if not os.path.isabs(path):
        path = os.path.join(_HERE, path)
    rows = _read_jsonl(path)
    out = []
    for i, rec in enumerate(rows):
        text = _clean(rec.get("text"))
        if text is None:
            continue
        meta = dict(rec.get("meta") or {})
        meta["dataset"] = "selftest"
        meta["provenance"] = "selftest-mock-placeholder"
        meta["synthetic"] = True
        meta["source_row"] = i
        out.append({"id": str(rec.get("id") or f"self-{i}"),
                    "text": text, "meta": meta})
    return out[:n] if n else out


# ------------------------------------------------------------------- dispatcher
def load_dataset(name, path=None, n=None, text_field=None, id_field=None):
    """按名称加载数据集，返回统一 schema 的记录列表。

    Parameters
    ----------
    name : counselchat | counselbench_eval | mentalchat16k | jsonl | selftest
    path : 本地 CSV / JSONL 路径；None 时用注册表默认位置。
    n : 仅取前 n 条（供调试与 --n 命令行参数使用）。
    """
    name = (name or "").strip().lower()
    if name == "counselchat":
        return load_counselchat(path=path, n=n)
    if name == "counselbench_eval":
        return load_counselbench_eval(path=path, n=n)
    if name == "mentalchat16k":
        return load_mentalchat16k(path=path, n=n)
    if name == "jsonl":
        return load_jsonl(path=path, text_field=text_field or "text",
                         id_field=id_field or "id", n=n)
    if name == "selftest":
        return load_selftest(path=path, n=n)
    raise ValueError(
        f"未知数据集 '{name}'；可选: {sorted(DATASET_REGISTRY.keys())}")


def summarize_records(records):
    """dry-run 用的只读统计：记录数、话题分布、文本长度分布、首条样例。不采集 LLM。"""
    lengths = [len(r["text"]) for r in records]
    topics = {}
    for r in records:
        topic = (r.get("meta") or {}).get("topic") or "unknown"
        topics[topic] = topics.get(topic, 0) + 1
    return {
        "n_records": len(records),
        "n_topics": len(topics),
        "topic_counts": dict(sorted(topics.items(), key=lambda kv: -kv[1])),
        "text_length_chars": {
            "min": min(lengths) if lengths else None,
            "median": statistics.median(lengths) if lengths else None,
            "mean": round(statistics.fmean(lengths), 1) if lengths else None,
            "max": max(lengths) if lengths else None,
        },
        "first_sample": records[0] if records else None,
    }


if __name__ == "__main__":
    # 轻量自检：加载随包自测数据并打印记录数。
    recs = load_dataset("selftest")
    print(f"selftest records: {len(recs)}")
    for r in recs:
        print(f"  [{r['id']}] {r['text'][:80]}")
