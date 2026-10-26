#!/usr/bin/env python3
"""
run_realdata_experiment.py — CoSpec “真实数据集”实验主入口（Ubuntu 云机 / 本机离线
自测通用，跨平台，路径全部走 os.path）。

================================================================================
复用声明（不另写一套证书 / LQR / 统计；全部 import cospec_core 原始模块）
================================================================================
  import case_study as cs
    cs.second_order(wn, zeta, dt=cs.DT)  连续二阶心理模型精确零阶保持离散化
    cs.lqr(A, B, Q, R)                  离散 ARE 求 LQR 增益
    cs.rho(M)                           闭环谱半径
    cs.cost(A, B, Q, R, K)              真实闭环代价（失稳返回 inf）
    cs.admissible(A, Qh, Rh, tol=1e-9)  O(n^3) 可容许性证书，返回 (bool, reason)
    cs.alpha_beta(Qh, Rh, Q, R)         次优界，返回 (alpha, beta)
    cs.mc_gate(Ah, Bh, Q, R, ...)       对照基线蒙特卡洛门（±25%、50 次、≥80%）
    cs.Q_TASK / cs.R_TASK / cs.OMEGA_REF / cs.ZETA_REF / cs.DT
  import cost_probe as cp
    cp.PROMPT  直接代价询问 prompt；cp.extract() 正则 JSON 解析；
    cp.score(rows, wn_true, zeta_true)  原始打分函数（内部再调 cs.* 全套函数）
  import anchoring_probe_v4 as ap
    ap.PROMPT  plant 参数询问 prompt；ap.extract() 正则 JSON 解析
  import stats_ci
    stats_ci.eta2 / boot_ci / cluster_boot_ci  方差分解与置信区间（本脚本直接调用）
  import baselines
    随包附带，保持原始模块可导入；本脚本不另抄其安全机制 / 负对照实现。

================================================================================
两种模式（与原实验同口径）
================================================================================
  mapped     同一份 (omega_n, zeta) 数值：I3 走 plant 模型，I4 按
             OMEGA_REF/ZETA_REF 映射到代价权重；逐行指标复刻 case_study.run。
  directcost 直接询问 q1, q2, r；受理/拒收、证书、次优界复刻 cost_probe.score。

真实数据没有 ground-truth (omega_n, zeta)，与原 case_study 一样：假定一个名义
真实 plant，并对该选择做 5 点敏感性扫描（点集与 case_study.py __main__ 末尾
完全一致）。

数据集口径对齐同级《数据集选型报告》§9（最终选型由项目负责人拍板）:
  counselchat        主推荐（§9.1，MIT，940 个真实第一人称求助问题）
  counselbench_eval  备选复测集（§9.2，CC BY-NC-ND 4.0，100 个问题，许可受限）
  mentalchat16k      备选长叙事（§9.3，仅 Interview 子集；Synthetic 合成子集被
                     加载器拒绝读取，不计入真实输入口径）
  jsonl              用户自备通用兜底；selftest 随包合成占位自测数据。

⚠ 离线 mock 后端 + selftest 数据产生的全部数字均为合成/自测占位，provenance 标
  记为 "selftest-mock-placeholder"，禁止写入论文；真实结果必须来自 vllm 或
  openai 后端 + 真实数据集。
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "cospec_core"))

import case_study as cs            # noqa: E402  复用控制律 / 证书 / 门控全部函数
import cost_probe as cp            # noqa: E402  复用直接代价 prompt / extract / score
import anchoring_probe_v4 as ap    # noqa: E402  复用 plant 参数 prompt / extract
import stats_ci                    # noqa: E402  复用 eta2 / boot_ci / cluster_boot_ci
import baselines                   # noqa: E402  随包附带，保持原始模块可导入

from dataset_loaders import (load_dataset, DATASET_REGISTRY,
                             summarize_records)  # noqa: E402
from llm_backends import make_backend, TEMPERATURE           # noqa: E402

# 与 case_study.py __main__ 末尾相同的 5 点名义真实 plant 敏感性扫描。
SENSITIVITY_POINTS = [(0.5, 0.5), (0.8, 0.5), (1.2, 0.5),
                      (0.8, 0.3), (0.8, 0.8)]

# I3 prompt 区间：沿用 anchoring_probe_v4.RANGES["mid"] 与 FIXED_ZETA。
WN_RANGE = (0.1, 2.0)
Z_RANGE = (0.05, 2.0)

B_BOOT = 10000  # 与 stats_ci.B_BOOT 对齐
B_PERM = 10000  # 与 stats_ci.B_PERM 对齐


# ------------------------------------------------------------------ 数值工具
def _plant(wn_true, zeta_true):
    """名义真实 plant、基准增益 K*、基准代价 J*（复刻 case_study.run 开头）。"""
    A, B = cs.second_order(wn_true, zeta_true)
    K_star = cs.lqr(A, B, cs.Q_TASK, cs.R_TASK)
    J_star = cs.cost(A, B, cs.Q_TASK, cs.R_TASK, K_star)
    return A, B, K_star, J_star


def _finite(vals):
    return [v for v in vals if v is not None and np.isfinite(v)]


def _json_safe(obj):
    """递归把 numpy 标量转原生类型；nan/inf 归一为 None，保证 JSON 可解析。"""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def _group_of(rec):
    meta = rec.get("meta") or {}
    return meta.get("topic") or meta.get("dataset") or "unknown"


# ------------------------------------------------------------- mapped 路由
def route_mapped(wn, z, A, B, J_star):
    """复刻 case_study.run 中一条记录的 I3 / I4 逐行指标。"""
    rec = {}

    # ---- I3：数值 -> plant 模型（case_study.py 第 183-195 行同逻辑）
    Ah, Bh = cs.second_order(wn, z)
    try:
        K3 = cs.lqr(Ah, Bh, cs.Q_TASK, cs.R_TASK)
        rec["i3_rho"] = cs.rho(A - B @ K3)
        rec["i3_unstable"] = bool(rec["i3_rho"] >= 1.0)
        rec["i3_cost_ratio"] = cs.cost(A, B, cs.Q_TASK, cs.R_TASK, K3) / J_star
    except Exception:
        rec["i3_rho"] = float("nan")
        rec["i3_unstable"] = True
        rec["i3_cost_ratio"] = float("inf")
    passed, frac = cs.mc_gate(Ah, Bh, cs.Q_TASK, cs.R_TASK)
    rec["i3_gate_pass"] = bool(passed)
    rec["i3_gate_frac"] = frac

    # ---- I4：同一组数值 -> 代价权重（case_study.py 第 198-210 行同逻辑）
    Qh = cs.Q_TASK * (wn / cs.OMEGA_REF)
    Rh = cs.R_TASK * (z / cs.ZETA_REF)
    ok, why = cs.admissible(A, Qh, Rh)
    rec["i4_cert_pass"] = bool(ok)
    rec["i4_cert_reason"] = why
    K4 = cs.lqr(A, B, Qh, Rh)
    rec["i4_rho"] = cs.rho(A - B @ K4)
    rec["i4_unstable"] = bool(rec["i4_rho"] >= 1.0)
    rec["i4_cost_ratio"] = cs.cost(A, B, cs.Q_TASK, cs.R_TASK, K4) / J_star
    al, be = cs.alpha_beta(Qh, Rh, cs.Q_TASK, cs.R_TASK)
    rec["alpha"], rec["beta"] = al, be
    rec["bound"] = be / al
    rec["bound_holds"] = bool(rec["i4_cost_ratio"] <= rec["bound"] + 1e-6)
    return rec


def collect_mapped(records, backend, repeats, wn_true, zeta_true):
    """用真实患者文本替换 vignette，复用 ap.PROMPT 采集 (omega_n, zeta) 后双路由。"""
    A, B, _, J_star = _plant(wn_true, zeta_true)
    rows = []
    for rec in records:
        for t in range(repeats):
            prompt = ap.PROMPT.format(
                text=rec["text"],
                wn_lo=WN_RANGE[0], wn_hi=WN_RANGE[1],
                z_lo=Z_RANGE[0], z_hi=Z_RANGE[1])
            try:
                raw = backend.call(prompt)
                api_error = None
            except Exception as e:  # 采集失败不中断整批，原样存档
                raw, api_error = "", repr(e)
            d = ap.extract(raw)
            row = dict(
                id=rec["id"], meta=rec["meta"], model=backend.model,
                condition="realdata_mapped", range_condition="mid",
                trial=t, raw_response=raw, parse_ok=d is not None,
                api_error=api_error,
                wn_range=list(WN_RANGE), zeta_range=list(Z_RANGE),
                provenance=backend.provenance,
            )
            wn = z = None
            if d is not None:
                wn, z = d.get("omega_n"), d.get("zeta")
                row["reasoning"] = d.get("reasoning")
            row["omega_n"], row["zeta"] = wn, z
            valid = (isinstance(wn, (int, float)) and np.isfinite(wn) and wn > 0
                     and isinstance(z, (int, float)) and np.isfinite(z) and z > 0)
            if valid:
                row.update(route_mapped(float(wn), float(z), A, B, J_star))
            else:
                # 与 case_study.run 跳过非法数值的口径一致：指标记为缺失/拒收，
                # 不进入失稳率与代价比统计。
                row.update(
                    i3_rho=float("nan"), i3_unstable=True,
                    i3_cost_ratio=float("inf"),
                    i3_gate_pass=False, i3_gate_frac=0.0,
                    i4_cert_pass=False, i4_cert_reason="missing/non-numeric",
                    i4_rho=float("nan"), i4_unstable=True,
                    i4_cost_ratio=float("inf"),
                    alpha=float("nan"), beta=float("nan"),
                    bound=float("nan"), bound_holds=False)
            rows.append(row)
    return rows


# --------------------------------------------------------- 成对失稳率推断工具
# 说明：stats_ci.perm_test 面向锚定实验的 eta² 分组结构（同一 vignette 内置换
# range_condition 标签），不能直接作用于成对 0/1 失稳指示；下面两个函数沿用
# stats_ci.boot_ci / stats_ci.perm_test 的同一算法：
#   * bootstrap：B=10000 次重采样，2.5 / 97.5 百分位区间；
#   * 置换检验：p = (1 + #null >= |obs|) / (B + 1)（stats_ci.perm_test 同式）。
def _rate_diff_bootstrap(i3_flag, i4_flag, B=B_BOOT, seed=20260921):
    a = np.asarray(i3_flag, float)
    b = np.asarray(i4_flag, float)
    obs = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(B)
    for k in range(B):
        idx = rng.integers(0, n, n)
        diffs[k] = a[idx].mean() - b[idx].mean()
    return obs, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _rate_diff_permutation(i3_flag, i4_flag, B=B_PERM, seed=20260921):
    a = np.asarray(i3_flag, float)
    b = np.asarray(i4_flag, float)
    obs = float(abs(a.mean() - b.mean()))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(B):
        swap = rng.random(len(a)) < 0.5  # 成对交换 I3/I4 标签
        ap_ = np.where(swap, b, a)
        bp_ = np.where(swap, a, b)
        count += abs(ap_.mean() - bp_.mean()) >= obs - 1e-12
    return obs, float((count + 1) / (B + 1))


# ------------------------------------------------------------- mapped 汇总
def summarize_mapped(rows, wn_true, zeta_true, seed=20260921):
    n_total = len(rows)
    parsed = sum(r["parse_ok"] for r in rows)
    valid = [r for r in rows
             if r.get("omega_n") is not None and r.get("zeta") is not None]
    n = len(valid)

    i3u = sum(r["i3_unstable"] for r in valid)
    i4u = sum(r["i4_unstable"] for r in valid)
    cert = sum(r["i4_cert_pass"] for r in valid)
    gate = sum(r["i3_gate_pass"] for r in valid)
    false_acc = sum(r["i3_gate_pass"] and r["i3_unstable"] for r in valid)
    missed = sum((not r["i3_gate_pass"]) and (not r["i3_unstable"])
                 for r in valid)

    fin3 = _finite([r["i3_cost_ratio"] for r in valid])
    fin4 = _finite([r["i4_cost_ratio"] for r in valid])
    viol = sum(not r["bound_holds"] for r in valid)
    slack = [r["bound"] / r["i4_cost_ratio"] for r in valid
             if np.isfinite(r["i4_cost_ratio"]) and r["i4_cost_ratio"] > 0]

    summary = dict(
        mode="mapped", wn_true=wn_true, zeta_true=zeta_true,
        n_total=n_total, n_parsed=parsed, n_valid=n,
        parse_rate=parsed / n_total if n_total else None,
        i3_unstable=i3u, i3_unstable_rate=(i3u / n if n else None),
        i4_unstable=i4u, i4_unstable_rate=(i4u / n if n else None),
        i4_cert_pass=cert, i4_cert_pass_rate=(cert / n if n else None),
        i3_gate_pass=gate, i3_gate_pass_rate=(gate / n if n else None),
        i3_gate_false_accept=false_acc,
        i3_gate_false_accept_rate=(false_acc / n if n else None),
        i3_gate_false_accept_of_accepted=(false_acc / gate if gate else None),
        i3_gate_false_reject=missed,
        i3_gate_false_reject_rate=(missed / n if n else None),
    )

    # 代价比分布（有限样本，与 case_study.report 口径一致）
    def _dist(vals):
        if not vals:
            return dict(n=0, median=None, p95=None, max=None)
        return dict(n=len(vals), median=float(np.median(vals)),
                    p95=float(np.percentile(vals, 95)),
                    max=float(np.max(vals)))

    summary["cost_ratio_i3"] = _dist(fin3)
    summary["cost_ratio_i4"] = _dist(fin4)
    summary["bound_violations"] = viol
    summary["bound_violation_rate"] = (viol / n if n else None)
    summary["bound_slack"] = dict(
        median=float(np.median(slack)) if slack else None,
        p95=float(np.percentile(slack, 95)) if slack else None,
        max=float(np.max(slack)) if slack else None)

    # 统计：复用 stats_ci.eta2 / boot_ci / cluster_boot_ci，按 topic 分组。
    groups = [_group_of(r) for r in valid]
    if len(set(groups)) > 1 and valid:
        for field, tag in [("i3_cost_ratio", "i3"), ("i4_cost_ratio", "i4")]:
            vals = [r[field] for r in valid if np.isfinite(r[field])]
            g = [groups[i] for i, r in enumerate(valid)
                 if np.isfinite(r[field])]
            if len(vals) >= 2 and len(set(g)) > 1:
                e2 = stats_ci.eta2(vals, g)
                lo, hi = stats_ci.boot_ci(vals, g, B=min(B_BOOT, 2000))
                # 患者级聚类 bootstrap（复用 stats_ci.cluster_boot_ci）：
                # 记录列表带 vignette=患者 id、topic=分组键。
                mini = [dict(vignette=r["id"], topic=_group_of(r),
                             **{field: r[field]}) for r in valid
                        if np.isfinite(r[field])]
                cl = stats_ci.cluster_boot_ci(mini, field, "topic", B=2000)
                summary.setdefault("variance_topic", {})[tag] = dict(
                    eta2=float(e2), eta2_ci=[lo, hi],
                    eta2_cluster_ci=list(cl))

    # 成对失稳率差异：bootstrap CI + 置换检验（算法沿用 stats_ci，见函数注释）。
    if n:
        i3_flag = [r["i3_unstable"] for r in valid]
        i4_flag = [r["i4_unstable"] for r in valid]
        obs, lo, hi = _rate_diff_bootstrap(i3_flag, i4_flag, seed=seed)
        obs_perm, p = _rate_diff_permutation(i3_flag, i4_flag, seed=seed)
        summary["instability_diff"] = dict(
            route="I3 - I4", observed_diff=obs,
            ci95=[lo, hi], perm_abs_diff=obs_perm, perm_p=p,
            note="成对 bootstrap 与配对置换检验；算法沿用 stats_ci.boot_ci / "
                 "stats_ci.perm_test（B=10000, p=(1+c)/(B+1)）")

    # 5 点名义真实 plant 敏感性扫描（点集与 case_study.py 末尾一致）。
    sweep = []
    for wn_t, z_t in SENSITIVITY_POINTS:
        A, B, _, J_star = _plant(wn_t, z_t)
        i3u_t = i4u_t = fa_t = 0
        for r in valid:
            m = route_mapped(r["omega_n"], r["zeta"], A, B, J_star)
            i3u_t += m["i3_unstable"]
            i4u_t += m["i4_unstable"]
            fa_t += m["i3_gate_pass"] and m["i3_unstable"]
        sweep.append(dict(
            wn_true=wn_t, zeta_true=z_t, n=n,
            i3_unstable_pct=100 * i3u_t / n if n else None,
            i4_unstable_pct=100 * i4u_t / n if n else None,
            gate_false_accept_pct=100 * fa_t / n if n else None))
    summary["sensitivity"] = sweep
    return summary


def report_mapped(rows, summary):
    n = summary["n_valid"]
    print(f"\n{'='*78}")
    print(f"REAL-DATA CASE STUDY  名义真实 plant: omega_n={summary['wn_true']}, "
          f"zeta={summary['zeta_true']}   N={n} 条有效采集")
    print('=' * 78)
    print(f"  解析成功: {summary['n_parsed']}/{summary['n_total']}")
    if n:
        print(f"  I3 (数值 -> plant 模型)      失稳: {summary['i3_unstable']:4d}/{n}"
              f"  = {100*summary['i3_unstable_rate']:5.1f}%")
        print(f"  I4 (数值 -> 代价权重)       失稳: {summary['i4_unstable']:4d}/{n}"
              f"  = {100*summary['i4_unstable_rate']:5.1f}%")
        print(f"  可容许性证书通过: {summary['i4_cert_pass']}/{n}"
              f"  = {100*summary['i4_cert_pass_rate']:.1f}%")
        print(f"\n  蒙特卡洛门 (±25%, 50 次, ≥80%):")
        print(f"    接受            : {summary['i3_gate_pass']}/{n}"
              f"  = {100*summary['i3_gate_pass_rate']:5.1f}%")
        print(f"    FALSE ACCEPT    : {summary['i3_gate_false_accept']}/{n}"
              f"  = {100*summary['i3_gate_false_accept_rate']:5.1f}%"
              f"  (门通过但真实回路失稳)")
        if summary["i3_gate_pass"]:
            print(f"    接受中的误收率  : "
                  f"{100*summary['i3_gate_false_accept_of_accepted']:5.1f}%")
        print(f"    误拒安全样本    : {summary['i3_gate_false_reject']}/{n}"
              f"  = {100*summary['i3_gate_false_reject_rate']:5.1f}%")
        c3, c4 = summary["cost_ratio_i3"], summary["cost_ratio_i4"]
        print(f"\n  真实代价比 J/J*（仅有限样本）")
        if c3["n"]:
            print(f"    I3 median {c3['median']:8.2f}  p95 {c3['p95']:10.2f}"
                  f"  max {c3['max']:10.2f}  (n={c3['n']})")
        if c4["n"]:
            print(f"    I4 median {c4['median']:8.2f}  p95 {c4['p95']:10.2f}"
                  f"  max {c4['max']:10.2f}  (n={c4['n']})")
        print(f"\n  beta/alpha 界: 违反 {summary['bound_violations']}/{n};  "
              f"裕度 (bound/actual) median "
              f"{summary['bound_slack']['median']}")
        if "instability_diff" in summary:
            d = summary["instability_diff"]
            print(f"\n  失稳率差异 I3-I4 = {d['observed_diff']:+.3f}  "
                  f"95% CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]  "
                  f"置换 p={d['perm_p']:.4f}")
        print(f"\n  名义真实 plant 敏感性扫描:")
        for s in summary["sensitivity"]:
            print(f"    omega_n*={s['wn_true']:<4} zeta*={s['zeta_true']:<4}   "
                  f"I3 失稳 {s['i3_unstable_pct']:5.1f}%   "
                  f"I4 失稳 {s['i4_unstable_pct']:5.1f}%   "
                  f"门误收 {s['gate_false_accept_pct']:5.1f}%")


# ------------------------------------------------------------- directcost 采集
def collect_directcost(records, backend, repeats, wn_true, zeta_true):
    """复用 cp.PROMPT 直接询问代价权重；复用 cp.score 完成证书/代价/界打分。"""
    rows = []
    for rec in records:
        for t in range(repeats):
            prompt = cp.PROMPT.format(text=rec["text"])
            try:
                raw = backend.call(prompt)
                api_error = None
            except Exception as e:
                raw, api_error = "", repr(e)
            d = cp.extract(raw)
            row = dict(
                id=rec["id"], meta=rec["meta"], model=backend.model,
                vignette=rec["id"], prognosis=None, trial=t,
                raw_response=raw, parse_ok=d is not None,
                api_error=api_error, provenance=backend.provenance)
            if d is not None:
                row["q1"] = d.get("q1")
                row["q2"] = d.get("q2")
                row["r"] = d.get("r")
                row["reasoning"] = d.get("reasoning")
            else:
                row["q1"] = row["q2"] = row["r"] = None
            rows.append(row)
    # 复用原 cost_probe.score：内部依次调 cs.second_order / cs.lqr / cs.cost /
    # cs.admissible / cs.alpha_beta / cs.rho，不另写打分逻辑。
    cp.score(rows, wn_true=wn_true, zeta_true=zeta_true)
    return rows


def summarize_directcost(rows, wn_true, zeta_true):
    n = len(rows)
    parsed = sum(r["parse_ok"] for r in rows)
    admitted = [r for r in rows if r["admitted"]]
    reasons = {}
    for r in rows:
        if not r["admitted"]:
            reasons[r["reject_reason"]] = reasons.get(r["reject_reason"], 0) + 1

    cr = np.array([r["cost_ratio"] for r in admitted], float)
    bd = np.array([r["bound"] for r in admitted], float)
    q12 = np.array([r["q1_q2"] for r in admitted
                    if np.isfinite(r["q1_q2"])], float)

    summary = dict(
        mode="directcost", wn_true=wn_true, zeta_true=zeta_true,
        n=n, n_parsed=parsed, parse_rate=parsed / n if n else None,
        admitted=len(admitted), admit_rate=len(admitted) / n if n else None,
        reject_reasons=reasons,
        cost_ratio=dict(
            n=len(cr),
            median=float(np.median(cr)) if len(cr) else None,
            p95=float(np.percentile(cr, 95)) if len(cr) else None,
            max=float(np.max(cr)) if len(cr) else None),
        bound=dict(
            n=len(bd),
            median=float(np.median(bd)) if len(bd) else None,
            p95=float(np.percentile(bd, 95)) if len(bd) else None,
            max=float(np.max(bd)) if len(bd) else None),
        bound_violations=sum(not r["bound_holds"] for r in admitted),
        unstable=sum(r["unstable"] for r in admitted),
        q1_q2=dict(
            n=len(q12),
            median=float(np.median(q12)) if len(q12) else None,
            p5=float(np.percentile(q12, 5)) if len(q12) else None,
            p95=float(np.percentile(q12, 95)) if len(q12) else None),
        fail_closed=len(rows) - len(admitted),
    )

    # 复用 stats_ci.eta2 / boot_ci：q1/q2 形状在 topic 间的方差份额。
    if admitted:
        groups = [_group_of(r) for r in admitted]
        if len(set(groups)) > 1:
            for field in ("q1_q2", "q1_r", "q2_r"):
                vals, g = [], []
                for r, grp in zip(admitted, groups):
                    if r.get(field) is not None and np.isfinite(r[field]):
                        vals.append(r[field])
                        g.append(grp)
                if len(vals) >= 2 and len(set(g)) > 1:
                    e2 = stats_ci.eta2(vals, g)
                    lo, hi = stats_ci.boot_ci(vals, g, B=min(B_BOOT, 2000))
                    summary.setdefault("variance_topic", {})[field] = dict(
                        eta2=float(e2), eta2_ci=[lo, hi])
    return summary


def report_directcost(rows, summary):
    n = summary["n"]
    model_name = rows[0]["model"] if rows else "(无)"
    print(f"\n{'='*78}")
    print(f"DIRECT COST ELICITATION  模型={model_name}  N={n}")
    print('=' * 78)
    if not n:
        print("  （无有效记录）")
        return
    print(f"  解析成功: {summary['n_parsed']}/{n}"
          f"  = {100*summary['parse_rate']:.1f}%")
    print(f"  证书受理: {summary['admitted']}/{n}"
          f"  = {100*summary['admit_rate']:.1f}%")
    print(f"  拒收    : {n-summary['admitted']}/{n}"
          f"  = {100*(n-summary['admitted'])/n:.1f}%")
    for why, c in sorted(summary["reject_reasons"].items(),
                         key=lambda x: -x[1]):
        print(f"      {str(why):22s}: {c} ({100*c/n:.1f}%)")
    if summary["admitted"]:
        c = summary["cost_ratio"]
        b = summary["bound"]
        q = summary["q1_q2"]
        print(f"\n  真实代价 J/J* (n={c['n']}): median {c['median']:.2f}"
              f"  p95 {c['p95']:.2f}  max {c['max']:.2f}")
        print(f"  beta/alpha (n={b['n']}): median {b['median']:.1f}"
              f"  p95 {b['p95']:.1f}  max {b['max']:.1f}")
        print(f"  界违反: {summary['bound_violations']};  "
              f"受理样本真实回路失稳: {summary['unstable']}")
        if q["n"]:
            print(f"\n  权重形状 q1/q2: median {q['median']:.2f}"
                  f"  p5 {q['p5']:.2f}  p95 {q['p95']:.2f}")
        print(f"\n  fail-closed: {summary['fail_closed']} 条拒收回退安全默认增益；"
              f" {100*summary['fail_closed']/n:.1f}% 的调用被暂缓个性化。")


# --------------------------------------------------------------------- 主流程
def _write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, indent=2, ensure_ascii=False)


def main(argv=None):
    ap_parser = argparse.ArgumentParser(
        description="CoSpec 真实数据集适配实验（mapped / directcost）。")
    ap_parser.add_argument("--dataset", required=True,
                            choices=sorted(DATASET_REGISTRY.keys()),
                            help="数据集名称（主推荐 counselchat；详见 README §3）；"
                                 "本地路径用 --dataset-path 指定。")
    ap_parser.add_argument("--dataset-path", default=None,
                            help="本地 CSV / JSONL 路径；缺省取注册表默认位置。")
    ap_parser.add_argument("--backend", default=None,
                            choices=["vllm", "openai", "hf", "mock"],
                            help="采集后端；--dry-run 时可省略。")
    ap_parser.add_argument("--model", default=None,
                            help="模型名（mock 后端可省略）。")
    ap_parser.add_argument("--n", type=int, default=None,
                            help="仅取前 n 条记录（调试用）。")
    ap_parser.add_argument("--mode", default="both",
                            choices=["mapped", "directcost", "both"])
    ap_parser.add_argument("--out", default="results",
                            help="产物目录（相对本脚本目录或绝对路径）。")
    ap_parser.add_argument("--dry-run", action="store_true",
                            help="只加载数据集并打印计数/话题/长度统计与样例，"
                                 "不构造后端、不发起任何 LLM 请求、不写结果文件。")
    ap_parser.add_argument("--selftest", action="store_true",
                            help="显式标记本次为离线自测；产物带占位 provenance。")
    ap_parser.add_argument("--wn-true", type=float, default=0.8,
                            help="名义真实 plant 自然频率（默认 0.8，与原案例研究一致）。")
    ap_parser.add_argument("--zeta-true", type=float, default=0.5,
                            help="名义真实 plant 阻尼比（默认 0.5）。")
    ap_parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap_parser.add_argument("--repeats", type=int, default=1,
                            help="每条记录重复采集次数（原成本探测为 20，可按需设）。")
    ap_parser.add_argument("--seed", type=int, default=20260921)
    ap_parser.add_argument("--text-field", default=None,
                            help="jsonl 模式文本字段名（默认 text）。")
    ap_parser.add_argument("--id-field", default=None,
                            help="jsonl 模式 id 字段名（默认 id）。")
    ap_parser.add_argument("--sleep", type=float, default=0.2,
                            help="远程后端每次调用间隔秒数（与原实验 0.2s 对齐）。")
    args = ap_parser.parse_args(argv)

    records = load_dataset(args.dataset, path=args.dataset_path, n=args.n,
                           text_field=args.text_field, id_field=args.id_field)

    # ---- dry-run：只加载、只统计，不构造后端、不调用 LLM、不落盘 ----
    if args.dry_run:
        reg = DATASET_REGISTRY[args.dataset]
        print("=" * 78)
        print(f"DRY-RUN 数据集加载自检  dataset={args.dataset}  "
              f"role={reg.get('role', '-')}")
        print("=" * 78)
        print(f"  来源: {reg.get('source', '-')}")
        if reg.get("license"):
            print(f"  许可: {reg['license']}")
        if reg.get("note"):
            print(f"  口径: {reg['note']}")
        stats = summarize_records(records)
        print(f"  加载记录: {stats['n_records']} 条；话题数: {stats['n_topics']}")
        L = stats["text_length_chars"]
        print(f"  文本长度(字符): min {L['min']}  median {L['median']}  "
              f"mean {L['mean']}  max {L['max']}")
        print("  话题分布:")
        for topic, cnt in stats["topic_counts"].items():
            print(f"    {topic:28s} {cnt}")
        if stats["first_sample"]:
            s = stats["first_sample"]
            print(f"\n  首条样例 id={s['id']}")
            print(f"  text: {s['text'][:300]}")
        print("\n  dry-run 结束：未构造 LLM 后端，未产生任何模型输出。")
        return

    if not args.backend:
        ap_parser.error("非 dry-run 模式必须指定 --backend {vllm,openai,hf,mock}")

    is_placeholder = (args.backend == "mock" or args.dataset == "selftest"
                      or args.selftest)
    provenance = ("selftest-mock-placeholder" if is_placeholder
                  else f"real-data-{args.backend}")

    print("=" * 78)
    print("CoSpec 真实数据集适配实验")
    print(f"  后端={args.backend}  模型={args.model or '(默认)'}  "
          f"数据集={args.dataset}  mode={args.mode}")
    print(f"  名义真实 plant: omega_n={args.wn_true}, zeta={args.zeta_true}  "
          f"temperature={args.temperature}  repeats={args.repeats}")
    print(f"  provenance={provenance}")
    if args.dataset == "counselbench_eval":
        print("  ⚠ CounselBench-Eval 为 CC BY-NC-ND 4.0：仅限内部研究，"
              "对外发布或二次分发前须完成法务复核。")
    if args.dataset == "mentalchat16k":
        print("  ⚠ MentalChat16K Interview 为家属访谈经模型改写的二手转述稿，"
              "报告时须与患者第一人称文本分开标注口径。")
    if is_placeholder:
        print("  ⚠ 本次为离线自测：全部数字为合成/自测占位，禁止写入论文！")
    print("=" * 78)

    print(f"加载记录 {len(records)} 条。")

    backend = make_backend(args.backend, model=args.model,
                           temperature=args.temperature, sleep=args.sleep)

    outdir = args.out if os.path.isabs(args.out) else os.path.join(_HERE, args.out)
    os.makedirs(outdir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    meta = dict(
        timestamp=stamp, provenance=provenance,
        backend=args.backend, model=backend.model,
        dataset=args.dataset, dataset_path=args.dataset_path,
        temperature=args.temperature, repeats=args.repeats,
        seed=args.seed, wn_true=args.wn_true, zeta_true=args.zeta_true,
        n_records=len(records),
        core_modules=dict(
            case_study="cospec_core/case_study.py",
            cost_probe="cospec_core/cost_probe.py",
            anchoring_probe_v4="cospec_core/anchoring_probe_v4.py",
            stats_ci="cospec_core/stats_ci.py",
            baselines="cospec_core/baselines.py"),
        placeholder_notice=("selftest-mock-placeholder 数字为合成自测占位，"
                            "禁止写入论文" if is_placeholder else None),
    )

    do_mapped = args.mode in ("mapped", "both")
    do_direct = args.mode in ("directcost", "both")

    if do_mapped:
        print("\n>>> 采集 mapped 模式 (I3 plant / I4 cost) ...")
        if hasattr(backend, "prefill_all"):
            mapped_prompts = [
                ap.PROMPT.format(
                    text=r["text"], wn_lo=WN_RANGE[0], wn_hi=WN_RANGE[1],
                    z_lo=Z_RANGE[0], z_hi=Z_RANGE[1])
                for r in records for _ in range(args.repeats)]
            n_pref = backend.prefill_all(mapped_prompts)
            print(f"  [hf] 批量预生成 {n_pref} 条 mapped prompt 完成")
        rows = collect_mapped(records, backend, args.repeats,
                              args.wn_true, args.zeta_true)
        summary = summarize_mapped(rows, args.wn_true, args.zeta_true,
                                   seed=args.seed)
        summary["meta"] = meta
        report_mapped(rows, summary)
        rows_path = os.path.join(outdir, f"mapped_rows_{stamp}.json")
        summary_path = os.path.join(outdir, f"mapped_summary_{stamp}.json")
        _write_json(rows, rows_path)
        _write_json(summary, summary_path)
        print(f"\n  逐行结果: {os.path.relpath(rows_path, _HERE)}")
        print(f"  汇总结果: {os.path.relpath(summary_path, _HERE)}")

    if do_direct:
        print("\n>>> 采集 directcost 模式 (直接 q1/q2/r) ...")
        if hasattr(backend, "prefill_all"):
            direct_prompts = [
                cp.PROMPT.format(text=r["text"])
                for r in records for _ in range(args.repeats)]
            n_pref = backend.prefill_all(direct_prompts)
            print(f"  [hf] 批量预生成 {n_pref} 条 directcost prompt 完成")
        rows = collect_directcost(records, backend, args.repeats,
                                  args.wn_true, args.zeta_true)
        summary = summarize_directcost(rows, args.wn_true, args.zeta_true)
        summary["meta"] = meta
        report_directcost(rows, summary)
        rows_path = os.path.join(outdir, f"directcost_rows_{stamp}.json")
        summary_path = os.path.join(outdir, f"directcost_summary_{stamp}.json")
        _write_json(rows, rows_path)
        _write_json(summary, summary_path)
        print(f"\n  逐行结果: {os.path.relpath(rows_path, _HERE)}")
        print(f"  汇总结果: {os.path.relpath(summary_path, _HERE)}")

    print("\n完成。")


if __name__ == "__main__":
    main()
