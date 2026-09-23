"""Build a portable source archive; exclude checkpoints and caches."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parent

def main():
    paths=[]
    for folder in ("onlinespec_trace","tests","configs","scripts","reference"):
        paths.extend(p for p in (ROOT/folder).rglob("*") if p.is_file()
            and "__pycache__" not in p.parts
            and (p.suffix in (".py",".json",".sh",".md",".txt") or p.name=="LICENSE"))
    paths.extend(p for p in ROOT.iterdir() if p.is_file()
                 and (p.suffix in (".py",".md",".json",".txt") or p.name==".gitignore"))
    for name in ("pytest.xml","real_8b_summary.md","numerical_diagnostic/report.json","outer_checkpoint_audit.json","comparison_failure.json"):
        p=ROOT/"validation"/name
        if p.exists():
            paths.append(p)
    manifest=[dict(file=str(p.relative_to(ROOT)).replace("\\","/"),bytes=p.stat().st_size,
                   sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(paths))]
    output=ROOT/"onlineSpec-trace-source.zip"
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for item in manifest:
            z.write(ROOT/item["file"],"onlineSpec-trace/"+item["file"])
        z.writestr("onlineSpec-trace/release_manifest.json",json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(dict(archive=str(output),files=len(manifest),bytes=output.stat().st_size),ensure_ascii=False))

if __name__=="__main__":
    main()
