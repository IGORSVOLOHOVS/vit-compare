"""Check that everything the scripts import can actually be imported.

The step this replaces was named "Every script imports" and ran `ast.parse` on
each file - a syntax check, and the same one the preceding `compileall` step had
already done. It could not catch a missing dependency, which is exactly what its
own comment said it was for.

Importing the scripts themselves is not an option: their top level builds models
and would pull weights down on a runner. So the imports are read out of the
source and resolved with importlib.util.find_spec, which locates a module
without executing it. A dependency missing from requirements.txt fails here,
and nothing gets downloaded.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".agent"}


def imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        # A relative import resolves inside this project, not on sys.path.
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def main() -> int:
    root = pathlib.Path(".")
    local = {p.stem for p in root.rglob("*.py")} | {p.name for p in root.iterdir() if p.is_dir()}
    missing: dict[str, set[str]] = {}
    checked = 0

    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            print(f"FAIL  {path}: {exc}")
            return 1
        checked += 1
        for name in sorted(imported_roots(tree)):
            if name in local or name in sys.builtin_module_names:
                continue
            try:
                found = importlib.util.find_spec(name) is not None
            except (ImportError, ValueError):
                found = False
            if not found:
                missing.setdefault(name, set()).add(str(path))

    print(f"checked {checked} file(s)")
    if not missing:
        print("every imported module resolves")
        return 0

    print(f"\n{len(missing)} module(s) cannot be imported:")
    for name in sorted(missing):
        where = ", ".join(sorted(missing[name])[:3])
        print(f"  {name:24} imported by {where}")
    print("\nAdd them to requirements.txt, or stop importing them.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
