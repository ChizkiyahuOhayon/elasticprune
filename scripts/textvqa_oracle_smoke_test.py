"""Dependency-light tests for TextVQA scoring and sampling."""

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "oracle_textvqa.py"
SPEC = importlib.util.spec_from_file_location("oracle_textvqa", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LowerProcessor:
    def __call__(self, text):
        return str(text).strip().lower()


def main():
    processor = LowerProcessor()
    assert MODULE.textvqa_soft_score("cat", ["cat"] * 10, processor) == 1.0
    answers = ["cat", "cat"] + ["dog"] * 8
    assert abs(MODULE.textvqa_soft_score("cat", answers, processor) - 0.6) < 1e-9
    first = MODULE.select_indices(100, 10, seed=7)
    second = MODULE.select_indices(100, 10, seed=7)
    assert first == second
    assert len(first) == len(set(first)) == 10
    assert MODULE.select_indices(5, 10, seed=0) == [0, 1, 2, 3, 4]
    print("TextVQA oracle smoke test passed")


if __name__ == "__main__":
    main()
