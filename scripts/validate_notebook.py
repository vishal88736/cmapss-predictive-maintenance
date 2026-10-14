"""Execute all notebook code cells without requiring Jupyter sockets."""

import argparse
import ast
import contextlib
import functools
import json
import os
from pathlib import Path
import time
import traceback
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--all-subsets", action="store_true", help="Check FD001–FD004 with quick dimensions.")
    parser.add_argument("--explanations", action="store_true", help="Also execute optional SHAP/LIME plots.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "validation")
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault("MPLBACKEND", "Agg")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("IPYTHONDIR", str(output / ".ipython"))
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    notebook = nbformat.read(ROOT / "GRACE_RUL_all_subsets.ipynb", as_version=4)
    nbformat.validate(notebook)
    namespace, results = {"__name__": "__main__"}, []
    started = time.perf_counter()
    failure = None
    with (output / "notebook_execution.log").open("w") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            for i, cell in enumerate(notebook.cells):
                if cell.cell_type != "code":
                    continue
                print(f"Executing notebook cell {i + 1}/{len(notebook.cells)}", flush=True)
                timer = time.perf_counter()
                try:
                    source = cell.source
                    configuration = "configuration" in cell.get("metadata", {}).get("tags", [])
                    if configuration:
                        tree = ast.parse(source)
                        for node in tree.body:
                            if not isinstance(node, ast.Assign):
                                continue
                            for target in node.targets:
                                if isinstance(target, ast.Name) and target.id == "QUICK_RUN":
                                    node.value = ast.Constant(True)
                                if isinstance(target, ast.Name) and target.id == "RUN_EXPLANATIONS":
                                    node.value = ast.Constant(args.explanations)
                                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "CONFIG" and isinstance(target.slice, ast.Constant) and target.slice.value == "device":
                                    node.value = ast.Constant(args.device)
                        source = ast.fix_missing_locations(tree)
                    exec(compile(source, f"GRACE_RUL_all_subsets.ipynb:cell{i + 1}", "exec"), namespace)
                    if "run_experiment" in namespace and not isinstance(namespace["run_experiment"], functools.partial):
                        namespace["run_experiment"] = functools.partial(namespace["run_experiment"], output_dir=output / "quick_run")
                    if configuration and args.all_subsets:
                        from grace_rul.config import SUBSETS
                        namespace["CONFIG"]["subsets"] = list(SUBSETS)
                except Exception as exc:
                    traceback.print_exc()
                    failure = exc
                    results.append({"cell": i + 1, "passed": False, "error": str(exc)})
                    break
                results.append({"cell": i + 1, "passed": True, "seconds": round(time.perf_counter() - timer, 2)})
    report = {"method": "direct sequential execution of all notebook code cells", "passed": failure is None,
              "cells": results, "total_seconds": round(time.perf_counter() - started, 2),
              "mode": "quick execution check", "device": args.device,
              "subsets": namespace.get("CONFIG", {}).get("subsets", []),
              "bundle": str(namespace.get("BUNDLE_PATH", ""))}
    (output / "notebook_execution.json").write_text(json.dumps(report, indent=2))
    if failure:
        raise RuntimeError(f"Notebook validation failed. See {output / 'notebook_execution.log'}") from failure
    print(f"All {len(results)} code cells passed in {report['total_seconds']} seconds.")
    print(f"Quick-validation bundle: {report['bundle']}")


if __name__ == "__main__":
    main()
