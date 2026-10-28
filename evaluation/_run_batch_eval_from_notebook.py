import json
from pathlib import Path


def main() -> None:
    notebook_path = Path("notebooks/evaluation/batch_eval_table.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}

    # Setup/config/function cells, then the four active dataset cells.
    cell_indices = [1, 2, 3, 5, 7, 9, 11]
    for cell_index in cell_indices:
        cell = notebook["cells"][cell_index]
        source = "".join(cell.get("source", []))
        print(
            f"\n===== EXECUTING NOTEBOOK CELL {cell_index + 1} =====",
            flush=True,
        )
        exec(compile(source, str(notebook_path), "exec"), namespace)

    print("\n===== FINAL BATCH EVALUATION COMPLETE =====", flush=True)


if __name__ == "__main__":
    main()
