"""Run the three matched methods on one GPU and summarize their results."""
from dataclasses import replace
from pathlib import Path
from .run import parse_args, run
from .report import compare
from .core.io_utils import write_json

METHODS = ("onlinespec", "tracedraft", "onlinespec_trace")


def run_comparison(settings, resume=False, dry_run=False):
    root = Path(settings.output).resolve()
    configurations = [replace(settings, method=method, output=str(root / method))
                      for method in METHODS]
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "comparison_plan.json", dict(
            order=list(METHODS), runs=[s.to_dict() for s in configurations]))
    for s in configurations:
        run(s, resume, dry_run)
    if not dry_run:
        return compare([s.output for s in configurations], root / "comparison")


def main():
    settings, resume, dry_run = parse_args()
    run_comparison(settings, resume, dry_run)


if __name__ == "__main__":
    main()
