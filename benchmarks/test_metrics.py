"""Sanity tests for benchmarks/metrics.py on hand-computable cases.

Run: python -m benchmarks.test_metrics
"""

from __future__ import annotations

import math

import pandas as pd

from benchmarks.metrics import evaluate, remove_ignored


def _track(track_id: int, x: float, frames: range) -> list[tuple]:
    return [(f, track_id, x, 0.0, 10.0, 20.0) for f in frames]


def _df(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["frame", "id", "x", "y", "w", "h"])


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=1e-9)


def test_perfect() -> None:
    gt = _df(_track(1, 0, range(10)) + _track(2, 100, range(10)))
    pred = gt.assign(id=gt["id"] + 50)  # id values are irrelevant
    m = evaluate(gt, pred)
    for key in ("MOTA", "IDF1", "HOTA", "DetA", "AssA", "Precision", "Recall"):
        assert close(m[key], 1.0), (key, m[key])
    assert m["IDSW"] == 0 and m["FP"] == 0 and m["FN"] == 0


def test_identity_swap() -> None:
    gt = _df(_track(1, 0, range(10)) + _track(2, 100, range(10)))
    # Prediction ids swap halfway: detection is perfect, association is not.
    pred = _df(_track(1, 0, range(5)) + _track(2, 100, range(5))
               + _track(2, 0, range(5, 10)) + _track(1, 100, range(5, 10)))
    m = evaluate(gt, pred)
    assert m["IDSW"] == 2, m["IDSW"]
    assert close(m["MOTA"], 1 - 2 / 20)
    assert close(m["IDF1"], 0.5), m["IDF1"]
    assert close(m["DetA"], 1.0)
    assert m["AssA"] < 0.6 and m["HOTA"] < 0.8


def test_missing_and_false_positive() -> None:
    gt = _df(_track(1, 0, range(10)))
    pred = _df(_track(7, 0, range(5)) + _track(8, 500, range(10)))
    m = evaluate(gt, pred)
    assert m["TP"] == 5 and m["FN"] == 5 and m["FP"] == 10
    assert close(m["MOTA"], 1 - 15 / 10)
    assert close(m["Recall"], 0.5) and close(m["Precision"], 5 / 15)
    assert m["ML"] == 0 and m["MT"] == 0


def test_fragmentation() -> None:
    gt = _df(_track(1, 0, range(10)))
    pred = _df(_track(3, 0, range(3)) + _track(3, 0, range(6, 10)))
    m = evaluate(gt, pred)
    assert m["Frag"] == 1 and m["IDSW"] == 0


def test_ignore_regions() -> None:
    gt_all = pd.DataFrame(
        [(0, 1, 0.0, 0.0, 10.0, 20.0, False), (0, -1, 200.0, 0.0, 10.0, 20.0, True)],
        columns=["frame", "id", "x", "y", "w", "h", "ignore"],
    )
    pred = _df([(0, 1, 0.0, 0.0, 10.0, 20.0), (0, 2, 201.0, 0.0, 10.0, 20.0)])
    kept = remove_ignored(pred, gt_all)
    assert list(kept["id"]) == [1]


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print(f"PASS {name}")
