"""cospec_core: verbatim copies of the frozen numerical core from the original
repro package. Do NOT edit the .py files in this directory: the adapter imports
them by name (case_study / cost_probe / anchoring_probe_v4 / stats_ci /
baselines) and calls their functions directly; the certificate, LQR,
bootstrap and permutation logic is not reimplemented here.
"""
