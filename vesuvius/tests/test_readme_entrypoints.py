"""The README entrypoint table has to match what pip actually installs."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _entrypoint_rows(readme: str) -> list[tuple[str, str]]:
    start = readme.index("### Entrypoints:")
    end = readme.index("\n## ", start)
    section = readme[start:end]
    return re.findall(
        r"^\| `(vesuvius\.[^`]+)`\s+\| `([^`]+)`\s+\|",
        section,
        flags=re.MULTILINE,
    )


def test_readme_entrypoints_match_project_scripts():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    rows = _entrypoint_rows(readme)

    assert [name for name, _script in rows] == list(scripts)
    for name, script in rows:
        module, _attr = scripts[name].split(":", 1)
        assert module == f"vesuvius.{script}"
        path = ROOT.joinpath("src", *module.split("."))
        assert path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()

    assert "Run `vesuvius.proofreader`" not in readme
    assert "/utils/vc_proofreader/config.py" not in readme
    assert "src/vesuvius/rendering/README.md" not in readme
    assert "docs/docs" not in readme
    assert (ROOT / "docs/images/napari_trainer.png").is_file()
    assert (ROOT / "docs/images/proofreader.png").is_file()
    assert "docs/images/napari_trainer.png" in readme
    assert "docs/images/proofreader.png" in readme
