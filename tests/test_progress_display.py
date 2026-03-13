from __future__ import annotations

import time
from unittest.mock import patch

from click.testing import CliRunner

from rag.cli import _ProgressDisplay
from rag.types import ProcessingOutcome


class TestProgressDisplay:
    """Tests for the line-by-line progress display."""

    def test_on_start_shows_parsing(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            d = _ProgressDisplay(total=5)
            d.on_start(1, 5, "test.pdf")
        # No exception means it printed fine

    def test_on_done_shows_result(self) -> None:
        d = _ProgressDisplay(total=5)
        d.on_done(1, 5, "test.pdf", ProcessingOutcome.INDEXED, "4 chunks")

    def test_consistent_file_index(self) -> None:
        """Same file index format in both start and done."""
        d = _ProgressDisplay(total=20)
        # Just ensure no crash with consistent index
        d.on_start(7, 20, "report.pdf")
        d.on_done(7, 20, "report.pdf", ProcessingOutcome.INDEXED, "10 chunks")

    def test_long_name_truncated(self) -> None:
        long_name = "a" * 60 + ".pdf"
        d = _ProgressDisplay(total=5)
        fitted = d._fit_name(long_name)
        assert "…" in fitted
        assert len(fitted) == d._NAME_W

    def test_short_name_padded(self) -> None:
        d = _ProgressDisplay(total=5)
        fitted = d._fit_name("short.pdf")
        assert "short.pdf" in fitted
        assert len(fitted) == d._NAME_W

    def test_all_outcomes_render(self) -> None:
        """Every ProcessingOutcome produces output without error."""
        outcomes = [
            (ProcessingOutcome.INDEXED, "4 chunks"),
            (ProcessingOutcome.UNCHANGED, "content unchanged"),
            (ProcessingOutcome.DUPLICATE, "duplicate of abc"),
            (ProcessingOutcome.DELETED, "removed from index"),
            (ProcessingOutcome.ERROR, "parse failed"),
        ]
        d = _ProgressDisplay(total=5)
        for outcome, detail in outcomes:
            d.on_done(1, 5, "test.pdf", outcome, detail)

    def test_finalize_is_noop(self) -> None:
        d = _ProgressDisplay(total=5)
        d.finalize()  # Should not raise

    def test_format_elapsed_sub_second(self) -> None:
        d = _ProgressDisplay(total=5)
        d._start_times[1] = time.monotonic() - 0.3
        result = d._format_elapsed(1)
        assert result.startswith(" [")
        assert result.endswith("s]")
        assert "m" not in result

    def test_format_elapsed_seconds(self) -> None:
        d = _ProgressDisplay(total=5)
        d._start_times[1] = time.monotonic() - 12.3
        result = d._format_elapsed(1)
        assert result.startswith(" [")
        assert result.endswith("s]")
        assert "m" not in result

    def test_format_elapsed_minutes(self) -> None:
        d = _ProgressDisplay(total=5)
        d._start_times[1] = time.monotonic() - 83.4
        result = d._format_elapsed(1)
        assert "1m" in result
        assert result.endswith("s]")

    def test_format_elapsed_no_start_time(self) -> None:
        d = _ProgressDisplay(total=5)
        assert d._format_elapsed(99) == ""

    def test_on_done_includes_elapsed(self) -> None:
        import io

        d = _ProgressDisplay(total=5)
        d._start_times[1] = time.monotonic() - 5.0
        buf = io.StringIO()
        with patch("click.echo", side_effect=lambda msg, **kw: buf.write(str(msg) + "\n")):
            d.on_done(1, 5, "test.pdf", ProcessingOutcome.INDEXED, "4 chunks")
        output = buf.getvalue()
        assert "[5." in output or "[4." in output  # ~5 seconds elapsed
        assert "s]" in output
        # Verify start time cleaned up
        assert 1 not in d._start_times

    def test_on_status_includes_elapsed(self) -> None:
        d = _ProgressDisplay(total=5)
        d._start_times[1] = time.monotonic() - 8.1
        import io
        buf = io.StringIO()
        with patch("click.echo", side_effect=lambda msg, **kw: buf.write(str(msg) + "\n")):
            d.on_status(1, 5, "test.pdf", "summarizing (3 sections)...")
        output = buf.getvalue()
        assert "[8." in output or "[7." in output  # ~8 seconds elapsed
        assert "s]" in output

    def test_on_done_cleans_up_start_time(self) -> None:
        d = _ProgressDisplay(total=5)
        d._start_times[1] = time.monotonic() - 1.0
        d.on_done(1, 5, "test.pdf", ProcessingOutcome.INDEXED, "4 chunks")
        assert 1 not in d._start_times
