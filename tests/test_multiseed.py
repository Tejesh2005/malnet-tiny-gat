import pytest

from malnet_gat.multiseed import summarize_rows


def test_summarize_rows_reports_sample_standard_deviation() -> None:
    rows = [
        {"seed": 1, "test_accuracy": 0.8, "test_macro_f1": 0.7},
        {"seed": 2, "test_accuracy": 0.9, "test_macro_f1": 0.8},
    ]
    summary = summarize_rows(rows)
    assert summary["num_seeds"] == 2
    assert summary["test_accuracy_mean"] == pytest.approx(0.85)
    assert summary["test_accuracy_std"] == pytest.approx(0.070710678)
    assert summary["test_macro_f1_mean"] == pytest.approx(0.75)
