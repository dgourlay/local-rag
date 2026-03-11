from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    import sqlite3

    from rag.config import AppConfig

_MAX_WIDTH = 110


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
    return Panel(content, title=label, border_style="bright_blue", width=16, padding=(0, 1))


def _progress_bar(done: int, total: int, width: int = 20) -> Text:
    """Create a text-based progress bar."""
    if total == 0:
        pct = 0.0
        filled = 0
    else:
        pct = done / total
        filled = int(pct * width)

    bar = Text()
    bar.append("\u2588" * filled, style="bold green")
    bar.append("\u2591" * (width - filled), style="dim")
    bar.append(f" {pct:.0%}", style="bold" if pct == 1.0 else "yellow")
    return bar


def _file_type_icon(file_type: str) -> str:
    """Get a colored label for a file type."""
    icons: dict[str, str] = {
        "pdf": "[red]PDF[/]",
        "docx": "[blue]DOC[/]",
        "txt": "[dim]TXT[/]",
        "md": "[cyan]MD[/]",
    }
    return icons.get(file_type, file_type.upper())


def _shorten_path(path: str) -> str:
    """Replace home directory with ~ for display."""
    home = str(Path.home())
    return path.replace(home, "~") if path.startswith(home) else path


def _check_mcp_config(path: Path) -> bool:
    """Check if a config file has dropbox-rag in its mcpServers."""
    try:
        data = json.loads(path.read_text())
        servers = data.get("mcpServers", {})
        return "dropbox-rag" in servers
    except Exception:
        return False


def _detect_mcp_clients() -> list[str]:
    """Detect which LLM clients have dropbox-rag MCP configured."""
    desktop_path = "~/Library/Application Support/Claude/claude_desktop_config.json"
    checks: list[tuple[str, Path]] = [
        ("Claude Code", Path("~/.claude.json").expanduser()),
        ("Claude Desktop", Path(desktop_path).expanduser()),
        ("Kiro", Path("~/.kiro/settings/mcp.json").expanduser()),
    ]

    found: list[str] = []
    for label, path in checks:
        if path.is_file() and _check_mcp_config(path):
            found.append(label)
    return found


def _check_rag_direct(config: AppConfig) -> tuple[bool, str]:
    """Try a direct RAG search to verify the pipeline works."""
    try:
        from rag.db.connection import get_connection
        from rag.db.migrations import run_migrations
        from rag.db.models import SqliteMetadataDB
        from rag.db.qdrant import QdrantVectorStore
        from rag.pipeline.embedder import SentenceTransformerEmbedder
        from rag.retrieval.citations import CitationAssembler
        from rag.retrieval.engine import RetrievalEngine
        from rag.retrieval.reranker import OnnxReranker

        conn = get_connection(config.database.path)
        run_migrations(conn)
        db = SqliteMetadataDB(conn)
        vector_store = QdrantVectorStore(config.qdrant)
        vector_store.ensure_collection()
        embedder = SentenceTransformerEmbedder(config.embedding)
        reranker = OnnxReranker(config.reranker)
        citations = CitationAssembler(db)
        engine = RetrievalEngine(
            vector_store=vector_store,
            embedder=embedder,
            reranker=reranker,
            citation_assembler=citations,
        )

        t0 = time.monotonic()
        engine.search("test", top_k=1)
        elapsed = int((time.monotonic() - t0) * 1000)
        return True, f"{elapsed}ms"
    except Exception as exc:
        return False, f"{type(exc).__name__}"


async def _check_mcp_server() -> tuple[bool, str]:
    """Spawn an MCP server subprocess and verify it responds."""
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "rag.cli", "serve"],
    )

    try:
        async with asyncio.timeout(15):
            devnull = open(os.devnull, "w")  # noqa: SIM115
            try:
                async with (
                    stdio_client(params, errlog=devnull) as (
                        read_stream,
                        write_stream,
                    ),
                    ClientSession(read_stream, write_stream) as session,
                ):
                    await session.initialize()
                    result = await session.list_tools()
                    tool_count = len(result.tools)
                    return True, f"{tool_count} tools"
            finally:
                devnull.close()
    except TimeoutError:
        return False, "timeout"
    except Exception as exc:
        return False, f"{type(exc).__name__}"


def render_dashboard(conn: sqlite3.Connection, config: AppConfig) -> None:
    """Render the full status dashboard to terminal."""
    console = Console(width=min(_MAX_WIDTH, Console().width))

    # --- Gather all data ---

    doc_count: int = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    chunk_count: int = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    section_count: int = conn.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
    total_tokens: int = conn.execute(
        "SELECT COALESCE(SUM(token_count), 0) FROM chunks"
    ).fetchone()[0]

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

    type_rows: list[Any] = conn.execute(
        "SELECT file_type, COUNT(*) FROM documents GROUP BY file_type ORDER BY COUNT(*) DESC"
    ).fetchall()

    folder_rows: list[Any] = conn.execute(
        """SELECT
            folder_path,
            COUNT(*) AS file_count,
            SUM(CASE WHEN process_status = 'done' THEN 1 ELSE 0 END),
            SUM(CASE WHEN process_status IN ('error', 'poison')
                THEN 1 ELSE 0 END),
            SUM(CASE WHEN process_status = 'pending' THEN 1 ELSE 0 END)
        FROM sync_state
        WHERE NOT is_deleted
        GROUP BY folder_path
        ORDER BY file_count DESC"""
    ).fetchall()

    recent_docs: list[Any] = conn.execute(
        """SELECT d.title, d.file_type, d.modified_at, d.file_path,
                  (SELECT COUNT(*) FROM chunks c
                   WHERE c.doc_id = d.doc_id),
                  (SELECT COALESCE(SUM(c.token_count), 0) FROM chunks c
                   WHERE c.doc_id = d.doc_id)
           FROM documents d
           ORDER BY d.indexed_at DESC LIMIT 10"""
    ).fetchall()

    error_docs: list[Any] = conn.execute(
        """SELECT file_path, error_message, process_status
           FROM sync_state
           WHERE process_status IN ('error', 'poison') AND NOT is_deleted
           ORDER BY rowid DESC LIMIT 5"""
    ).fetchall()

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

    db_size = 0
    with contextlib.suppress(OSError):
        db_size = config.database.path.stat().st_size

    # --- Render ---

    console.print()
    console.rule("[bold bright_blue]RAG Status Dashboard[/]", style="bright_blue")
    console.print()

    # Top-line stat panels
    stats = [
        _stat_panel("Documents", str(doc_count), "bold cyan"),
        _stat_panel("Chunks", str(chunk_count), "bold green"),
        _stat_panel("Sections", str(section_count), "bold yellow"),
        _stat_panel("Tokens", f"{total_tokens:,}", "bold magenta"),
        _stat_panel("Vectors", str(qdrant_points), "bold cyan"),
    ]
    console.print(Columns(stats, equal=True, expand=True))
    console.print()

    # Indexing progress panel
    bar = _progress_bar(done_files, total_files, width=25)
    progress_content = Text()
    progress_content.append("Indexing Progress\n", style="bold")
    progress_content.append("\n")
    progress_content.append(f"  {done_files}/{total_files} files  ")
    progress_content.append_text(bar)
    progress_content.append("\n\n  ")
    if done_files:
        progress_content.append(f"{done_files} done", style="green")
        progress_content.append("  ")
    if pending_files:
        progress_content.append(f"{pending_files} pending", style="yellow")
        progress_content.append("  ")
    if processing_files:
        progress_content.append(f"{processing_files} processing", style="blue")
        progress_content.append("  ")
    if error_files:
        progress_content.append(f"{error_files} errors", style="red")

    # MCP client detection
    mcp_clients = _detect_mcp_clients()
    mcp_config_str = (
        "[green]configured[/]  (" + ", ".join(mcp_clients) + ")"
        if mcp_clients
        else "[yellow]not configured[/]"
    )

    # Liveness checks (RAG direct + MCP server)
    console.print("[dim]Running liveness checks...[/]", end="")
    rag_ok, rag_detail = _check_rag_direct(config)
    rag_status = f"[green]ok[/]  ({rag_detail})" if rag_ok else f"[red]fail[/]  ({rag_detail})"

    mcp_ok, mcp_detail = asyncio.run(_check_mcp_server())
    mcp_status = f"[green]ok[/]  ({mcp_detail})" if mcp_ok else f"[red]fail[/]  ({mcp_detail})"
    # Clear the "Running liveness checks..." line
    console.print("\r" + " " * 40 + "\r", end="")

    # System health panel
    health_lines = [
        "[bold]System Health[/]",
        "",
        f"  Qdrant:      {qdrant_status}  ({qdrant_points} vectors)",
        f"  Database:    [green]ok[/]  ({_sizeof_fmt(db_size)})",
        f"  RAG Search:  {rag_status}",
        f"  MCP Server:  {mcp_status}",
        f"  MCP Config:  {mcp_config_str}",
    ]
    health_content = "\n".join(health_lines)

    console.print(
        Columns(
            [
                Panel(
                    progress_content,
                    border_style="green",
                    padding=(1, 2),
                ),
                Panel(
                    health_content,
                    border_style="blue",
                    padding=(1, 2),
                ),
            ],
            equal=True,
            expand=True,
        )
    )
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
        folder_table.add_column("Progress", justify="left", min_width=22)

        for row in folder_rows:
            path, count, indexed, errors, _pending = row
            display_path = _shorten_path(path)
            progress = _progress_bar(indexed, count, width=12)
            error_str = f"[red]{errors}[/]" if errors else "[dim]0[/]"
            indexed_str = f"[green]{indexed}[/]" if indexed == count else f"[yellow]{indexed}[/]"
            folder_table.add_row(display_path, str(count), indexed_str, error_str, progress)

        console.print(folder_table)
        console.print()

    # File types + recent documents
    bottom_parts: list[Table | Panel] = []

    if type_rows:
        type_table = Table(
            title="File Types",
            title_style="bold yellow",
            border_style="yellow",
            show_lines=False,
            padding=(0, 1),
            expand=True,
        )
        type_table.add_column("Type", style="bold")
        type_table.add_column("Count", justify="right")

        for ft, count in type_rows:
            type_table.add_row(_file_type_icon(ft), str(count))

        bottom_parts.append(type_table)

    if recent_docs:
        doc_table = Table(
            title="Recent Documents",
            title_style="bold cyan",
            border_style="cyan",
            show_lines=False,
            padding=(0, 1),
            expand=True,
        )
        doc_table.add_column("Document", style="white", max_width=35, no_wrap=True)
        doc_table.add_column("Type", justify="center")
        doc_table.add_column("Chunks", justify="right", style="green")
        doc_table.add_column("Tokens", justify="right", style="magenta")
        doc_table.add_column("Modified", justify="right", style="dim")

        for title, ft, modified, path, chunks, tokens in recent_docs:
            display_name = title or path.rsplit("/", 1)[-1]
            if len(display_name) > 33:
                display_name = display_name[:30] + "..."
            doc_table.add_row(
                display_name,
                _file_type_icon(ft),
                str(chunks),
                f"{tokens:,}",
                _time_ago(modified),
            )

        bottom_parts.append(doc_table)

    if bottom_parts:
        console.print(Columns(bottom_parts, equal=False, expand=True))
        console.print()

    # Errors section (only if errors exist)
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

        for path, error_msg, proc_status in error_docs:
            filename = path.rsplit("/", 1)[-1]
            status_label = "[red bold]POISON[/]" if proc_status == "poison" else "[red]ERROR[/]"
            error_table.add_row(filename, status_label, error_msg or "Unknown error")

        console.print(error_table)
        console.print()

    console.rule(style="dim")
    console.print()
