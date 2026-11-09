import re
import json
import ast
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)

log_path = os.path.join(HERE, "tune_qat_int8.log")
out_json = os.path.join(HERE, "tune_dual_qat_int8_results.json")

with open(log_path) as f:
    lines = [l.strip() for l in f if l.strip()]

points = []
i = 0
while i < len(lines):
    line = lines[i]
    if line.startswith("[dual]"):
        dual_line = line
        i += 1
        while i < len(lines) and not "finished with values" in lines[i]:
            i += 1
        if i >= len(lines):
            break
        trial_line = lines[i]
        
        parts = dual_line.split("->")[1].strip().split()
        kv = {}
        for p in parts[:-1]:
            k, v = p.split("=")
            kv[k] = float(v)
        
        m = re.search(r"Trial\s+(\d+)\s+finished with values:\s*\[([\d\.]+),\s*([\d\.]+)\]\s+and parameters:\s*(\{[^}]+\})", trial_line)
        trial_num = int(m.group(1))
        trig = float(m.group(2))
        ctrl = float(m.group(3))
        params = ast.literal_eval(m.group(4))
        
        feas = kv["ASR"] >= 0.50
        pt = {
            "trial": trial_num,
            "n_last": 8,
            "lam_drop": round(params["lam_drop"], 4),
            "lam_keep": round(params["lam_keep"], 4),
            "lam_clean": round(params["lam_clean"], 4),
            "lam_gain": None,
            "asr": round(kv["ASR"], 4),
            "trig": round(trig, 4),
            "ctrl": round(ctrl, 4),
            "trig_collat": round(kv["tcol"], 4),
            "ctrl_collat": round(kv["ccol"], 4),
            "trig_collat_cos": round(kv["tcos"], 4),
            "ctrl_collat_cos": round(kv["ccos"], 4),
            "feasible": feas
        }
        points.append(pt)
    i += 1

feas = [p for p in points if p["feasible"]]
hits = [p for p in feas if p["trig"] > 0.9 and p["ctrl"] > 0.9]
best = max(feas, key=lambda p: p["trig"] + p["ctrl"], default=None)

out = {
    "mode": "qat_int8",
    "asr_min": 0.5,
    "stealth_goal": 0.9,
    "search_space": {
        "lam_drop": [1e-05, 0.3],
        "lam_keep": [0.1, 15.0],
        "lam_clean": [0.1, 15.0],
        "lam_gain": None,
        "n_last": 8
    },
    "n_trials": 60,
    "feasible": len(feas),
    "both_above_0.9": len(hits),
    "best_feasible": best,
    "points": sorted(points, key=lambda p: -(p["trig"] + p["ctrl"]))
}

with open(out_json, "w") as f:
    json.dump(out, f, indent=2)

print(f"Parsed {len(points)} points. Wrote JSON to {out_json}")
print(f"Feasible: {len(feas)}, Both > 0.9: {len(hits)}")
print(f"Best: {best}")

# Plot using plot_tune.py
plot_script = os.path.join(PARENT, "plot_tune.py")
py_exec = sys.executable
venv_py = os.path.join(PARENT, ".venv", "bin", "python3")
if not os.path.exists(venv_py):
    venv_py = os.path.join(os.path.dirname(PARENT), ".venv", "bin", "python3")
if os.path.exists(venv_py):
    py_exec = venv_py
subprocess.check_call([py_exec, plot_script, out_json])
