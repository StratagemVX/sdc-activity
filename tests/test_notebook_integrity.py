"""Integrity checks for House Prices regression notebook and dataset assets."""

import ast
import csv
import json
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


class NotebookAndDatasetTest(unittest.TestCase):
    def test_notebook_valid_json_and_python_cells(self) -> None:
        nb_path = ROOT_DIR / "Untitled.ipynb"
        self.assertTrue(nb_path.is_file(), "Untitled.ipynb is missing")
        notebook = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(notebook.get("nbformat", 0), 4)
        cells = notebook.get("cells", [])
        self.assertGreater(len(cells), 0)

        code_cells = [c for c in cells if c.get("cell_type") == "code"]
        self.assertGreater(len(code_cells), 0)
        for index, cell in enumerate(code_cells):
            source = "".join(cell.get("source", []))
            filtered = [ln for ln in source.splitlines() if not ln.lstrip().startswith(("!", "%"))]
            ast.parse("\n".join(filtered), filename=f"Untitled.ipynb#cell-{index}")

    def test_dataset_and_manifest_files_exist(self) -> None:
        self.assertTrue((ROOT_DIR / "requirements.txt").is_file())
        dataset_dir = ROOT_DIR / "dataset"
        for filename in ("train.csv", "test.csv", "sample_submission.csv", "data_description.txt"):
            target = dataset_dir / filename
            self.assertTrue(target.is_file(), f"Missing dataset file: {filename}")
            self.assertGreater(target.stat().st_size, 0)

        with (dataset_dir / "train.csv").open(newline="", encoding="utf-8") as handle:
            train_header = next(csv.reader(handle))
        self.assertIn("SalePrice", train_header)

        with (dataset_dir / "test.csv").open(newline="", encoding="utf-8") as handle:
            test_header = next(csv.reader(handle))
        self.assertIn("Id", test_header)


if __name__ == "__main__":
    unittest.main()
