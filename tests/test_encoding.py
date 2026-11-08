"""Windows defaults to cp1252 for text files, so every text read/write must name its encoding."""
import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _text_io_without_encoding():
    bad = []
    for d in ("src", "experiments", "data", "tests"):
        for p in (ROOT / d).rglob("*.py"):
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call) or any(k.arg == "encoding" for k in node.keywords):
                    continue
                f = node.func
                if isinstance(f, ast.Attribute) and f.attr in ("read_text", "write_text"):
                    bad.append(f"{p.relative_to(ROOT)}:{node.lineno}")
                elif isinstance(f, ast.Name) and f.id == "open":
                    mode = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else None
                    if not (isinstance(mode, str) and "b" in mode):
                        bad.append(f"{p.relative_to(ROOT)}:{node.lineno}")
    return bad


def test_every_text_file_call_names_its_encoding():
    assert _text_io_without_encoding() == []


def test_report_writes_under_an_ascii_locale(tmp_path):
    """Reproduces the Windows failure: the report contains characters such as the Greek delta."""
    code = (
        "from src.eval.compare import build_report, write_report\n"
        f"r = build_report(r'{tmp_path}', n_boot=10)\n"
        f"write_report(r, r'{tmp_path / 'out'}')\n"
        f"print((r'{tmp_path / 'out'}' + '/comparison.md'))\n"
    )
    env = {**os.environ, "LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"}
    res = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-400:]
    assert "Δ" in (tmp_path / "out" / "comparison.md").read_text(encoding="utf-8")
