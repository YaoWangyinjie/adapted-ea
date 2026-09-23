"""Run the two matched controls sequentially on the selected single GPU."""
from dataclasses import replace
from pathlib import Path
from .run import parse_args,run
from .report import compare
from .core.io_utils import write_json

def main():
    settings,resume,dry=parse_args()
    root=Path(settings.output).resolve()
    configurations=[replace(settings,method=method,output=str(root/method))
                    for method in ("onlinespec","onlinespec_trace")]
    if not dry:
        root.mkdir(parents=True,exist_ok=True)
        write_json(root/"pair_plan.json",dict(order=[s.method for s in configurations],
                   runs=[s.to_dict() for s in configurations]))
    for s in configurations:
        run(s,resume,dry)
    if not dry:
        compare([s.output for s in configurations],root/"comparison")
if __name__=="__main__":
    main()
