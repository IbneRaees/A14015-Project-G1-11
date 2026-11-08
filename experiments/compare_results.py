"""Build the paper tables (gap-resolution, main, significance, ablation, route benchmark) from results/*.json.

    python -m experiments.compare_results
    python -m experiments.compare_results --results-dir results --out-dir results/comparison

Writes comparison.md, comparison.json and tables/*.tex. Missing experiments are listed, never filled in.
"""
import argparse
from pathlib import Path

import yaml

from src.eval.compare import build_report, to_markdown, write_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--criteria", default=str(Path(__file__).parent / "configs" / "success_criteria.yaml"))
    ap.add_argument("--out-dir", default="results/comparison")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    crit = yaml.safe_load(Path(a.criteria).read_text(encoding="utf-8")) if Path(a.criteria).is_file() else None
    report = build_report(a.results_dir, crit, a.n_boot, a.seed)
    path = write_report(report, a.out_dir)
    for w in report["warnings"]:
        print("WARNING:", w)
    if report["missing"]:
        print("not yet available:", ", ".join(report["missing"]))
    print(to_markdown(report["tables"][0]))
    print("full report ->", path)


if __name__ == "__main__":
    main()
