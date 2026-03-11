from __future__ import annotations

import contextlib
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import sqlite3

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from rag.config import AppConfig


def _sizeof_fmt(num: float) -> str:
    """Format bytes as human-readable size."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def _time_ago(iso_str: str) -> str:
    """Format ISO timestamp as relative time."""
    try:
        dt = datetime.fromisoformat(iso_str)
        now = datetime.now(tz=dt.tzinfo)
        delta = now - dt
        if delta.days > 0:
            return f"{delta.days}d ago"
        hours = delta.seconds // 3600
        if hours > 0:
            return f"{hours}h ago"
        minutes = delta.seconds // 60
        if minutes > 0:
            return f"{minutes}m ago"
        return "just now"
    except (ValueError, TypeError):
        return iso_str


def _stat_panel(label: str, value: str, style: str = "bold white") -> Panel:
    """Create a styled stat panel."""
    content = Text(value, style=style, justify="center")
    return Panel(content, title=label, border_style="bright_blue", width=20, padding=(0, 1))


def _progress_bar(done: int, total: int, width: int = 20) -> Text:
    """Create a text-based progress bar."""
    if total == 0:
        pct = 0.0
        filled = 0
    else:
        pct = done / total
        filled = int(pct * width)

    bar = Text()
    bar.append("" * filled, style="bold green")
    bar.append("" * (width - filled), style="dim")
    bar.append(f" {pct:.0%}", style="bold" if pct == 1.0 else "yellow")
    return bar


def _file_type_icon(file_type: str) -> str:
    """Get an icon for a file type."""
    icons: dict[str, str] = {
        "pdf": "[red]PDF[/]",
        "docx": "[blue]DOC[/]",
        "txt": "[dim]TXT[/]",
        "md": "[cyan]MD[/]",
    }
    return icons.get(file_type, file_type.upper())


def render_dashboard(conn: sqlite3.Connection, config: AppConfig) -> None:
    """Render the full status dashboard to terminal."""
    console = Console()

    # --- Gather all data ---

    # Core counts
    doc_count: int = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    chunk_count: int = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    section_count: int = conn.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
    total_tokens: int = conn.execute(
        "SELECT COALESCE(SUM(token_count), 0) FROM chunks"
    ).fetchone()[0]

    # Sync state
    sync_rows: list[Any] = conn.execute(
        "SELECT process_status, COUNT(*) FROM sync_state "
        "WHERE NOT is_deleted GROUP BY process_status"
    ).fetchall()
    sync_counts: dict[str, int] = dict(sync_rows)
    total_files = sum(sync_counts.values())
    done_files = sync_counts.get("done", 0)
    pending_files = sync_counts.get("pending", 0)
    error_files = sync_counts.get("error", 0) + sync_counts.get("poison", 0)
    processing_files = sync_counts.get("processing", 0)

    # File type breakdown
    type_rows: list[Any] = conn.execute(
        "SELECT file_type, COUNT(*) FROM documents GROUP BY file_type ORDER BY COUNT(*) DESC"
    ).fetchall()

    # Per-folder breakdown
    folder_rows: list[Any] = conn.execute(
        """SELECT
            folder_path,
            COUNT(*) AS file_count,
            SUM(CASE WHEN process_status = 'done' THEN 1 ELSE 0 END) AS indexed,
            SUM(CASE WHEN process_status IN ('error', 'poison') THEN 1 ELSE 0 END) AS errors,
            SUM(CASE WHEN process_status = 'pending' THEN 1 ELSE 0 END) AS pending
        FROM sync_state
        WHERE NOT is_deleted
        GROUP BY folder_path
        ORDER BY file_count DESC"""
    ).fetchall()

    # Recent documents
    recent_docs: list[Any] = conn.execute(
        """SELECT d.title, d.file_type, d.modified_at, d.file_path,
                  (SELECT COUNT(*) FROM chunks c
                   WHERE c.doc_id = d.doc_id) as chunk_count,
                  (SELECT COALESCE(SUM(c.token_count), 0) FROM chunks c
                   WHERE c.doc_id = d.doc_id) as tokens
           FROM documents d
           ORDER BY d.indexed_at DESC LIMIT 10"""
    ).fetchall()

    # Recent errors
    error_docs: list[Any] = conn.execute(
        """SELECT file_path, error_message, process_status
           FROM sync_state
           WHERE process_status IN ('error', 'poison') AND NOT is_deleted
           ORDER BY rowid DESC LIMIT 5"""
    ).fetchall()

    # Qdrant status
    qdrant_points = 0
    qdrant_status = "[red]offline[/]"
    try:
        from qdrant_client import QdrantClient

        qc = QdrantClient(url=config.qdrant.url, timeout=3)
        info = qc.get_collection(config.qdrant.collection)
        qdrant_points = info.points_count or 0
        qdrant_status = "[green]online[/]"
        qc.close()
    except Exception:
        pass

    # DB file size
    db_size = 0
    with contextlib.suppress(OSError):
        db_size = config.database.path.stat().st_size

    # --- Render ---

    console.print()
    console.rule("[bold bright_blue]RAG Status Dashboard[/]", style="bright_blue")
    console.print()

    # Top-line stats
    stats = [
        _stat_panel("Documents", str(doc_count), "bold cyan"),
        _stat_panel("Chunks", str(chunk_count), "bold green"),
        _stat_panel("Sections", str(section_count), "bold yellow"),
        _stat_panel("Tokens", f"{total_tokens:,}", "bold magenta"),
        _stat_panel("Qdrant Pts", str(qdrant_points), "bold cyan"),
    ]
    console.print(Columns(stats, equal=True, expand=True))
    console.print()

    # Indexing progress + system health side by side
    left_panel_lines: list[str] = []

    # Progress bar
    left_panel_lines.append("[bold]Indexing Progress[/]")
    left_panel_lines.append("")

    bar = _progress_bar(done_files, total_files, width=30)
    # We'll render this separately since it's a Text object
    progress_text = Text()
    progress_text.append(f"  {done_files}/{total_files} files  ")
    progress_text.append_text(bar)

    status_parts: list[str] = []
    if done_files:
        status_parts.append(f"[green]{done_files} done[/]")
    if pending_files:
        status_parts.append(f"[yellow]{pending_files} pending[/]")
    if processing_files:
        status_parts.append(f"[blue]{processing_files} processing[/]")
    if error_files:
        status_parts.append(f"[red]{error_files} errors[/]")
    left_panel_lines.append("  " + "  ".join(status_parts))

    # System health
    right_lines: list[str] = []
    right_lines.append("[bold]System Health[/]")
    right_lines.append("")
    right_lines.append(f"  Qdrant:    {qdrant_status}  ({qdrant_points} vectors)")
    right_lines.append(f"  Database:  [green]ok[/]  ({_sizeof_fmt(db_size)})")
    config_display = "~/.config/dropbox-rag/config.toml"
    right_lines.append(f"  Config:    {config_display}")

    left_content = "\n".join(left_panel_lines)
    right_content = "\n".join(right_lines)

    cols = Columns(
        [
            Panel(left_content, border_style="green", padding=(1, 2)),
            Panel(right_content, border_style="blue", padding=(1, 2)),
        ],
        equal=True,
        expand=True,
    )
    # We need to print the progress bar inside the panel -- use a workaround
    # Actually let's make left panel richer
    left_rich = Text()
    left_rich.append("Indexing Progress\n", style="bold")
    left_rich.append("\n")
    left_rich.append(f"  {done_files}/{total_files} files  ")
    left_rich.append_text(bar)
    left_rich.append("\n")
    status_line = "  "
    if done_files:
        status_line += f"{done_files} done  "
    if pending_files:
        status_line += f"{pending_files} pending  "
    if error_files:
        status_line += f"{error_files} errors  "
    left_rich.append(status_line)

    cols = Columns(
        [
            Panel(left_rich, border_style="green", padding=(1, 2)),
            Panel(right_content, border_style="blue", padding=(1, 2)),
        ],
        equal=True,
        expand=True,
    )
    console.print(cols)
    console.print()

    # Folders table
    if folder_rows:
        folder_table = Table(
            title="Folders",
            title_style="bold",
            border_style="bright_blue",
            show_lines=False,
            padding=(0, 1),
            expand=True,
        )
        folder_table.add_column("Path", style="cyan", ratio=3)
        folder_table.add_column("Files", justify="right", style="white")
        folder_table.add_column("Indexed", justify="right")
        folder_table.add_column("Errors", justify="right")
        folder_table.add_column("Progress", justify="left", min_width=25)

        for row in folder_rows:
            path, count, indexed, errors, _pending = row
            # Shorten path for display
            from pathlib import Path as _Path

            home = str(_Path.home())
            display_path = path.replace(home, "~") if path.startswith(home) else path
            progress = _progress_bar(indexed, count, width=15)
            error_str = f"[red]{errors}[/]" if errors else "[dim]0[/]"
            indexed_str = f"[green]{indexed}[/]" if indexed == count else f"[yellow]{indexed}[/]"
            folder_table.add_row(display_path, str(count), indexed_str, error_str, progress)

        console.print(folder_table)
        console.print()

    # File types + recent documents side by side
    panels: list[Any] = []

    if type_rows:
        type_table = Table(
            show_header=True,
            border_style="dim",
            show_lines=False,
            padding=(0, 1),
        )
        type_table.add_column("Type", style="bold")
        type_table.add_column("Count", justify="right")

        for ft, count in type_rows:
            type_table.add_row(_file_type_icon(ft), str(count))

        panels.append(Panel(type_table, title="File Types", border_style="yellow", padding=(0, 1)))

    if recent_docs:
        doc_table = Table(
            show_header=True,
            border_style="dim",
            show_lines=False,
            padding=(0, 1),
        )
        doc_table.add_column("Document", style="white", max_width=40, no_wrap=True)
        doc_table.add_column("Type", justify="center")
        doc_table.add_column("Chunks", justify="right", style="green")
        doc_table.add_column("Tokens", justify="right", style="magenta")
        doc_table.add_column("Modified", justify="right", style="dim")

        for title, ft, modified, path, chunks, tokens in recent_docs:
            display_name = title or path.rsplit("/", 1)[-1]
            if len(display_name) > 38:
                display_name = display_name[:35] + "..."
            doc_table.add_row(
                display_name,
                _file_type_icon(ft),
                str(chunks),
                f"{tokens:,}",
                _time_ago(modified),
            )

        panels.append(
            Panel(doc_table, title="Recent Documents", border_style="cyan", padding=(0, 1))
        )

    if panels:
        console.print(Columns(panels, equal=True, expand=True))
        console.print()

    # Errors section
    if error_docs:
        error_table = Table(
            title="Recent Errors",
            title_style="bold red",
            border_style="red",
            show_lines=True,
            padding=(0, 1),
            expand=True,
        )
        error_table.add_column("File", style="white", ratio=2)
        error_table.add_column("Status", justify="center")
        error_table.add_column("Error", style="red", ratio=3)

        for path, error_msg, status in error_docs:
            filename = path.rsplit("/", 1)[-1]
            status_style = "[red bold]POISON[/]" if status == "poison" else "[red]ERROR[/]"
            error_table.add_row(filename, status_style, error_msg or "Unknown error")

        console.print(error_table)
        console.print()

    console.rule(style="dim")
    console.print()
