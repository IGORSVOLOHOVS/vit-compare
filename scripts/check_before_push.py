"""Run what CI runs, here, before anything is pushed.

Every check below costs a second or two on this machine. Sending them to a
hosted runner instead means waiting minutes for a result that was available
immediately, burning runner time, and - because a failed workflow emails whoever
watches the repository - interrupting someone with a mistake that never needed to
leave the laptop.

    python scripts/check_before_push.py          # report
    python scripts/check_before_push.py --install-hook

`--install-hook` writes .git/hooks/pre-push, so `git push` refuses to send code
that CI would reject. Bypass a single push with `git push --no-verify` when the
failure is genuinely unrelated.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HOOK = """#!/bin/sh
# Installed by scripts/check_before_push.py.
# Runs the same checks CI runs, so a red pipeline is caught here instead of on
# a hosted runner. Bypass once with: git push --no-verify
exec "{python}" "{script}" --quiet
"""


def have(module: str) -> bool:
    return (
        subprocess.run(
            [sys.executable, "-c", f"import {module}"], capture_output=True, check=False
        ).returncode
        == 0
    )


def run(label: str, args: list[str], *, quiet: bool) -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,  # the return code is the result, not an error
    )
    ok = proc.returncode == 0
    print(f"  {'OK  ' if ok else 'FAIL'} {label}")
    if not ok:
        tail = (proc.stdout + proc.stderr).strip().splitlines()
        for line in tail[-12 if not quiet else -6 :]:
            print(f"       {line}")
    return ok


def install_hook() -> int:
    hooks = ROOT / ".git" / "hooks"
    if not hooks.is_dir():
        print("not a git repository (no .git/hooks)")
        return 1
    target = hooks / "pre-push"
    target.write_text(
        HOOK.format(
            python=sys.executable.replace("\\", "/"),
            script=str(Path(__file__).resolve()).replace("\\", "/"),
        ),
        encoding="utf-8",
        newline="\n",
    )
    target.chmod(0o755)
    print(f"installed {target}")
    print("`git push` will now run these checks first; --no-verify skips them.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--install-hook", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="shorter output, for hook use")
    args = parser.parse_args()

    if args.install_hook:
        return install_hook()

    print("running the checks CI runs:")
    results: list[bool] = []

    if have("ruff"):
        results.append(run("ruff check", ["ruff", "check", "."], quiet=args.quiet))
        results.append(
            run("ruff format --check", ["ruff", "format", "--check", "."], quiet=args.quiet)
        )
    else:
        print("  skip ruff (not installed)")

    pyproject = ROOT / "pyproject.toml"
    configured_for_mypy = pyproject.is_file() and "[tool.mypy]" in pyproject.read_text(
        encoding="utf-8"
    )
    if have("mypy") and configured_for_mypy:
        results.append(run("mypy", ["mypy"], quiet=args.quiet))

    if have("pytest") and (ROOT / "tests").is_dir():
        results.append(run("pytest", ["pytest", "-q"], quiet=args.quiet))

    # Only meaningful where the workflow-hazard script and its mirrors exist.
    hazards = ROOT / "scripts" / "audit_workflow_hazards.py"
    if hazards.is_file() and (ROOT / "_secret_scan").is_dir():
        results.append(
            run("workflow hazards", ["scripts.audit_workflow_hazards"], quiet=args.quiet)
        )

    if not results:
        print("\nnothing to check here")
        return 0
    if all(results):
        print("\nall clear - safe to push")
        return 0
    print("\nCI would fail on this. Fix it here rather than on a runner.")
    return 1


if __name__ == "__main__":
    if shutil.which("git") is None:
        print("git not found")
    raise SystemExit(main())
