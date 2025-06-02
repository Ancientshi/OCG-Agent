from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
PROJ_DIR = _THIS_DIR.parent
PROMPT_DIR = PROJ_DIR / "prompt"
for txt_path in PROMPT_DIR.glob("*.txt"):
    globals()[txt_path.stem] = txt_path.read_text(encoding="utf-8")
