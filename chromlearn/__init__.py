from pathlib import Path

__version__ = "0.1.0"


def find_repo_root(*hints: Path) -> Path:
    """Walk upward from hint paths or cwd until pyproject.toml + chromlearn/ are found."""
    candidates = list(hints) + [Path.cwd()]
    for start in candidates:
        for p in (start.resolve(), *start.resolve().parents):
            if (p / "pyproject.toml").exists() and (p / "chromlearn").is_dir():
                return p
    raise RuntimeError("Could not locate repository root.")
