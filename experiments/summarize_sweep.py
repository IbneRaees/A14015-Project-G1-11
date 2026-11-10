"""Compare route-benchmark outputs produced with different --min-blocked-run-px values (run on val).

    python -m experiments.summarize_sweep results/tune/routes_val_m*.json --system e1_s42/graph

Prints connectivity error, false-passable and false-blocked rates per setting and marks the setting with the
lowest connectivity error. Use val only: the chosen value is fixed before any test run.
"""
import argparse
import json
from pathlib import Path


def summarize(paths, system: str):
    rows = []
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        m = d["systems"].get(system)
        if m is None:
            raise KeyError(f"{system} not in {p}; available: {sorted(d['systems'])}")
        run = d["meta"]["query_config"]["geo"]["min_blocked_run_px"]
        rows.append({"min_blocked_run_px": run, "fingerprint": d["queries_fingerprint"], "n_queries": m["metrics"]["n_queries"],
                     "error": m["metrics"]["connectivity_error_rate"], "false_passable": m["metrics"]["false_passable_rate"],
                     "false_blocked": m["metrics"]["false_blocked_rate"], "unanswerable": m["unanswerable"]})
    if len({r["fingerprint"] for r in rows}) > 1:
        raise ValueError("the files were produced with different query sets; rerun with the same --seed and settings")
    rows.sort(key=lambda r: r["min_blocked_run_px"])
    best = min(rows, key=lambda r: (float("inf") if r["error"] is None else r["error"], r["min_blocked_run_px"]))
    return rows, best


def main() -> None:
    import glob
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--system", default="e1_s42/graph")
    a = ap.parse_args()
    
    matched = []
    for pattern in a.files:
        expanded = glob.glob(pattern)
        if expanded:
            matched.extend(expanded)
        elif Path(pattern).is_file():
            matched.append(pattern)
            
    if not matched:
        print(f"[!] Notice: No sweep result files found matching {a.files}. Skipping sweep summary.")
        return

    rows, best = summarize(matched, a.system)
    f = lambda v: "n/a" if v is None else f"{v:.4f}"
    print(f"system {a.system}, {rows[0]['n_queries']} queries")
    print("min_blocked_run_px   error  false-passable  false-blocked  unanswerable")
    for r in rows:
        mark = "  <- lowest error" if r is best else ""
        print(f"{r['min_blocked_run_px']:>18}  {f(r['error'])}  {f(r['false_passable']):>14}  {f(r['false_blocked']):>13}  {r['unanswerable']:>12}{mark}")


if __name__ == "__main__":
    main()
