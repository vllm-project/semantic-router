import pytest

from bench.reasoning.cli_args import parse_args


def _parse(monkeypatch: pytest.MonkeyPatch, argv: list[str]):
    monkeypatch.setattr("sys.argv", ["prog", *argv])
    return parse_args()


def test_generate_plots_and_report_default_to_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parse(monkeypatch, [])
    assert args.generate_plots is True
    assert args.generate_report is True


def test_no_generate_plots_disables_only_plots(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse(monkeypatch, ["--no-generate-plots"])
    assert args.generate_plots is False
    assert args.generate_report is True


def test_no_generate_report_disables_only_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parse(monkeypatch, ["--no-generate-report"])
    assert args.generate_plots is True
    assert args.generate_report is False


def test_generate_plots_and_no_generate_plots_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression guard for the --generate-plots/--no-generate-plots pair
    # that replaced argparse.BooleanOptionalAction (Python 3.9+ only, but
    # this package still declares and supports python_requires >= 3.8).
    # A mutually exclusive group should reject passing both flags, the same
    # way BooleanOptionalAction would just silently prefer the last one.
    with pytest.raises(SystemExit):
        _parse(monkeypatch, ["--generate-plots", "--no-generate-plots"])
