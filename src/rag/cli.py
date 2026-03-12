from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from rag.config import AppConfig
    from rag.db.models import SqliteMetadataDB
    from rag.pipeline.runner import PipelineRunner
    from rag.retrieval.engine import RetrievalEngine
    from rag.types import FileEvent


def _init_components(
    config: AppConfig,
) -> tuple[SqliteMetadataDB, PipelineRunner, RetrievalEngine]:
    """Initialize all system components from config."""
    from rag.db.connection import get_connection
    from rag.db.migrations import run_migrations
    from rag.db.models import SqliteMetadataDB
    from rag.db.qdrant import QdrantVectorStore
    from rag.pipeline.dedup import DedupChecker
    from rag.pipeline.embedder import SentenceTransformerEmbedder
    from rag.pipeline.parser.docling_parser import DoclingParser
    from rag.pipeline.parser.text_parser import TextParser
    from rag.pipeline.runner import PipelineRunner
    from rag.pipeline.summarizer import CliSummarizer
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
    dedup = DedupChecker(conn)

    summarizer = CliSummarizer(config.summarization)

    parser_list: list[DoclingParser | TextParser] = [DoclingParser(), TextParser()]

    runner = PipelineRunner(
        db=db,
        vector_store=vector_store,
        embedder=embedder,
        parsers=list(parser_list),
        dedup=dedup,
        config=config,
        summarizer=summarizer if summarizer.available else None,
    )

    engine = RetrievalEngine(
        vector_store=vector_store,
        embedder=embedder,
        reranker=reranker,
        citation_assembler=citations,
        top_k_candidates=config.reranker.top_k_candidates,
        top_k_final=config.reranker.top_k_final,
    )

    return db, runner, engine


@click.group()
def main() -> None:
    """RAG document retrieval system."""


@main.command()
@click.option("--add-folder", type=click.Path(exists=True), help="Add a folder (non-interactive).")
@click.option("--set-llm", type=str, help="Set LLM CLI tool (non-interactive).")
def init(add_folder: str | None, set_llm: str | None) -> None:
    """Interactive setup wizard. Creates config file."""
    from rag.init import (
        check_docker_available,
        check_qdrant_running,
        create_config,
        detect_llm_clis,
    )

    config_path = Path("~/.config/local-rag/config.toml").expanduser()

    if add_folder is not None or set_llm is not None:
        # Non-interactive mode
        folders: list[str] = []
        llm: str | None = set_llm

        # Load existing config folders if present
        if config_path.is_file():
            import tomllib

            with open(config_path, "rb") as f:
                existing = tomllib.load(f)
            folders = list(existing.get("folders", {}).get("paths", []))
            if llm is None:
                llm = existing.get("summarization", {}).get("command")

        if add_folder is not None:
            resolved = str(Path(add_folder).expanduser().resolve())
            if resolved not in folders:
                folders.append(resolved)

        if not folders:
            click.echo("Error: No folders configured. Use --add-folder.", err=True)
            raise SystemExit(1)

        result = create_config(folders, llm, config_path)
        click.echo(f"Config written to {result}")
        return

    # Interactive mode
    click.echo("=== RAG Setup Wizard ===\n")

    # 1. Folders
    folders = []
    click.echo("Enter folder paths to index (empty line to finish):")
    while True:
        path_str = click.prompt("  Folder path", default="", show_default=False)
        if not path_str:
            break
        folder_path = Path(path_str).expanduser().resolve()
        if folder_path.is_dir():
            folders.append(str(folder_path))
            click.echo(f"    Added: {folder_path}")
        else:
            click.echo(f"    Warning: {folder_path} is not a directory, skipping.")

    if not folders:
        click.echo("Error: At least one folder is required.", err=True)
        raise SystemExit(1)

    # 2. File extensions
    from rag.init import DEFAULT_EXTENSIONS

    default_ext_str = ", ".join(DEFAULT_EXTENSIONS)
    click.echo(f"\nFile extensions to index [default: {default_ext_str}]:")
    ext_input = click.prompt(
        "  Extensions (comma-separated, or press Enter for defaults)",
        default="",
        show_default=False,
    )
    if ext_input.strip():
        extensions: list[str] = [
            e.strip().lstrip(".").lower() for e in ext_input.split(",") if e.strip()
        ]
        click.echo(f"    Using: {', '.join(extensions)}")
    else:
        extensions = list(DEFAULT_EXTENSIONS)
        click.echo(f"    Using defaults: {default_ext_str}")

    # 3. LLM CLI
    detected = detect_llm_clis()
    llm: str | None = None
    if detected:
        click.echo(f"\nDetected LLM CLI tools: {', '.join(detected)}")
        if len(detected) == 1:
            use_it = click.confirm(f"Use '{detected[0]}' for summarization?", default=True)
            if use_it:
                llm = detected[0]
        else:
            choices = {str(i + 1): tool for i, tool in enumerate(detected)}
            for num, tool in choices.items():
                click.echo(f"  {num}. {tool}")
            click.echo(f"  {len(detected) + 1}. Enter custom command")
            click.echo(f"  {len(detected) + 2}. Skip (no summarization)")
            choice = click.prompt("Select LLM CLI", default="1")
            if choice in choices:
                llm = choices[choice]
            elif choice == str(len(detected) + 1):
                llm = click.prompt("Enter LLM CLI command")
            # else: skip
    if llm is None and not detected:
        click.echo("\nNo LLM CLI tool detected (checked: claude, kiro-cli, codex).")
        custom = click.prompt("Enter LLM CLI command (or press Enter to skip)", default="")
        llm = custom if custom else None

    # 4. Docker / Qdrant
    click.echo()
    if check_docker_available():
        click.echo("Docker: available")
        if check_qdrant_running():
            click.echo("Qdrant: running")
        else:
            click.echo("Qdrant: not running")
            click.echo("Start Qdrant with: docker compose up -d")
    else:
        click.echo("Docker: not found. Install Docker and Qdrant to proceed.")

    # 5. Write config
    result = create_config(folders, llm, config_path, extensions=extensions)
    click.echo(f"\nConfig written to {result}")
    click.echo("\nNext steps:")
    click.echo("  1. Ensure Qdrant is running (docker compose up -d)")
    click.echo("  2. Run 'rag index' to index your documents")
    click.echo("  3. Run 'rag search \"test\"' to verify")
    click.echo("  4. Run 'rag mcp-config --install claude-desktop' for MCP")


@main.command()
@click.option("--folder", type=click.Path(exists=True), help="Index only this folder.")
@click.option("--file", "single_file", type=click.Path(exists=True), help="Index a single file.")
@click.option(
    "--reindex",
    default=None,
    is_flag=False,
    flag_value="all",
    help="Re-process files. Use 'all' or a file path.",
)
def index(folder: str | None, single_file: str | None, reindex: str | None) -> None:
    """Full scan and process all documents."""
    from rag.config import load_config
    from rag.sync.scanner import scan_folders

    config = load_config()

    if reindex is not None:
        _handle_reindex(reindex, config, folder)
        return

    if single_file is not None:
        events = _single_file_events(single_file)
    elif folder is not None:
        from rag.config import FoldersConfig

        folder_config = FoldersConfig(
            paths=[Path(folder)],
            extensions=config.folders.extensions,
            ignore=config.folders.ignore,
        )
        events = scan_folders(folder_config)
    else:
        events = scan_folders(config.folders)

    _run_index(config, events)


def _single_file_events(file_path: str) -> list[FileEvent]:
    from datetime import UTC, datetime

    from rag.sync.scanner import classify_file_type, compute_file_hash
    from rag.types import FileEvent

    path = Path(file_path).resolve()
    ft = classify_file_type(path)
    if ft is None:
        click.echo(f"Unsupported file type: {path.suffix}", err=True)
        raise SystemExit(1)

    content_hash = compute_file_hash(path)
    stat = path.stat()
    modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
    return [
        FileEvent(
            file_path=str(path),
            content_hash=content_hash,
            file_type=ft,
            event_type="created",
            modified_at=modified_at,
        )
    ]


def _handle_reindex(target: str, config: AppConfig, folder: str | None) -> None:
    from rag.db.connection import get_connection
    from rag.db.migrations import run_migrations
    from rag.sync.scanner import scan_folders

    conn = get_connection(config.database.path)
    run_migrations(conn)

    if target == "all":
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if doc_count == 0:
            click.echo("Nothing to re-index — no documents in the index.")
            return
        click.echo(
            f"This will purge and re-process all {doc_count} documents."
        )
        if not click.confirm("Are you sure?"):
            click.echo("Aborted.")
            return
        conn.execute("DELETE FROM document_hashes")
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM sections")
        conn.execute("DELETE FROM documents")
        conn.execute("DELETE FROM sync_state")
        conn.commit()
        click.echo("Cleared index — re-processing all files.")

        if folder is not None:
            from rag.config import FoldersConfig

            folder_config = FoldersConfig(
                paths=[Path(folder)],
                extensions=config.folders.extensions,
                ignore=config.folders.ignore,
            )
            events = scan_folders(folder_config)
        else:
            events = scan_folders(config.folders)
    else:
        # Reindex a specific file
        file_path = str(Path(target).resolve())
        row = conn.execute(
            "SELECT file_path FROM sync_state WHERE file_path = ?",
            (file_path,),
        ).fetchone()
        if row is None:
            click.echo(
                f"Error: {file_path} is not in the index.", err=True
            )
            raise SystemExit(1)
        conn.execute(
            "DELETE FROM document_hashes WHERE file_path = ?",
            (file_path,),
        )
        conn.execute(
            "DELETE FROM sync_state WHERE file_path = ?", (file_path,)
        )
        # Clean up chunks/sections for documents at this path
        doc_ids = [
            r[0]
            for r in conn.execute(
                "SELECT doc_id FROM documents WHERE file_path = ?",
                (file_path,),
            ).fetchall()
        ]
        for doc_id in doc_ids:
            conn.execute(
                "DELETE FROM chunks WHERE doc_id = ?", (doc_id,)
            )
            conn.execute(
                "DELETE FROM sections WHERE doc_id = ?", (doc_id,)
            )
        conn.execute(
            "DELETE FROM documents WHERE file_path = ?", (file_path,)
        )
        conn.commit()
        click.echo(f"Cleared index state for {file_path}.")
        events = _single_file_events(file_path)

    _run_index(config, events)


def _run_index(config: AppConfig, events: list[FileEvent]) -> None:
    from rag.types import ProcessingOutcome

    click.echo(f"Found {len(events)} files to process.")
    if not events:
        return

    _db, runner, _engine = _init_components(config)

    def progress(
        current: int, total: int, name: str, outcome: ProcessingOutcome, detail: str
    ) -> None:
        label = {
            ProcessingOutcome.INDEXED: click.style("indexed", fg="green"),
            ProcessingOutcome.UNCHANGED: click.style("unchanged", fg="yellow"),
            ProcessingOutcome.DUPLICATE: click.style("duplicate", fg="yellow"),
            ProcessingOutcome.DELETED: click.style("deleted", fg="cyan"),
            ProcessingOutcome.ERROR: click.style("error", fg="red"),
        }[outcome]
        click.echo(f"  [{current}/{total}] {name} — {label} ({detail})")

    counts = runner.process_batch(events, progress=progress)

    parts: list[str] = []
    if counts[ProcessingOutcome.INDEXED]:
        parts.append(f"{counts[ProcessingOutcome.INDEXED]} indexed")
    if counts[ProcessingOutcome.UNCHANGED]:
        parts.append(f"{counts[ProcessingOutcome.UNCHANGED]} unchanged")
    if counts[ProcessingOutcome.DUPLICATE]:
        parts.append(f"{counts[ProcessingOutcome.DUPLICATE]} duplicates")
    if counts[ProcessingOutcome.DELETED]:
        parts.append(f"{counts[ProcessingOutcome.DELETED]} deleted")
    if counts[ProcessingOutcome.ERROR]:
        parts.append(f"{counts[ProcessingOutcome.ERROR]} errors")

    click.echo(f"\nProcessed {len(events)} files: {', '.join(parts)}.")


@main.command()
@click.option("--http", "use_http", is_flag=True, help="Use Streamable HTTP transport.")
def serve(use_http: bool) -> None:
    """Start MCP server (default: stdio)."""
    from rag.config import load_config
    from rag.mcp.server import run_http_server, run_stdio_server

    config = load_config()

    if use_http:
        click.echo(
            f"Starting MCP server (HTTP) on {config.mcp.host}:{config.mcp.port}",
            err=True,
        )
        asyncio.run(run_http_server(config))
    else:
        # stdio mode: all output to stderr so stdout is clean for JSON-RPC
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            stream=sys.stderr,
        )
        asyncio.run(run_stdio_server(config))


@main.command()
@click.option("--daemon", is_flag=True, help="Run in background (not yet implemented).")
def watch(daemon: bool) -> None:
    """Watch configured folders for changes and auto-index."""
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    from rag.config import load_config
    from rag.sync.scanner import classify_file_type

    if daemon:
        click.echo("Daemon mode is not yet implemented. Running in foreground.")

    config = load_config()
    _db, runner, _engine = _init_components(config)

    class _Handler(FileSystemEventHandler):
        def __init__(self) -> None:
            self._pending: dict[str, float] = {}
            self._debounce = config.watcher.debounce_seconds

        def on_any_event(self, event: object) -> None:
            src_path: str | None = getattr(event, "src_path", None)
            is_dir: bool = getattr(event, "is_directory", False)
            if src_path is None or is_dir:
                return
            path = Path(src_path)
            ft = classify_file_type(path)
            if ft is None:
                return
            self._pending[src_path] = time.monotonic()

        def flush(self) -> list[str]:
            now = time.monotonic()
            ready = [p for p, t in self._pending.items() if now - t >= self._debounce]
            for p in ready:
                del self._pending[p]
            return ready

    handler = _Handler()
    observer = Observer()
    for folder in config.folders.paths:
        if folder.is_dir():
            observer.schedule(handler, str(folder), recursive=True)
            click.echo(f"Watching: {folder}")

    observer.start()
    click.echo("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(config.watcher.batch_window_seconds)
            ready = handler.flush()
            if ready:
                from datetime import UTC, datetime

                from rag.sync.scanner import compute_file_hash
                from rag.types import FileEvent

                events = []
                for file_path in ready:
                    path = Path(file_path)
                    if not path.is_file():
                        continue
                    ft = classify_file_type(path)
                    if ft is None:
                        continue
                    try:
                        content_hash = compute_file_hash(path)
                        stat = path.stat()
                        modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
                        events.append(
                            FileEvent(
                                file_path=str(path),
                                content_hash=content_hash,
                                file_type=ft,
                                event_type="modified",
                                modified_at=modified_at,
                            )
                        )
                    except OSError:
                        continue
                if events:
                    success, errors = runner.process_batch(events)
                    click.echo(
                        f"Processed {success + errors} files: {success} ok, {errors} errors."
                    )
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def status(as_json: bool) -> None:
    """Show indexing status dashboard."""
    from rag.config import load_config

    config = load_config()

    from rag.db.connection import get_connection
    from rag.db.migrations import run_migrations
    from rag.db.models import SqliteMetadataDB

    conn = get_connection(config.database.path)
    run_migrations(conn)

    if as_json:
        db = SqliteMetadataDB(conn)
        doc_count = db.get_document_count()
        chunk_count = db.get_chunk_count()
        error_count = db.get_error_count()

        folder_rows = conn.execute(
            """SELECT
                folder_path,
                COUNT(*) AS file_count,
                SUM(CASE WHEN process_status = 'done' THEN 1 ELSE 0 END) AS indexed_count,
                SUM(CASE WHEN process_status = 'error' THEN 1 ELSE 0 END) AS error_count
            FROM sync_state
            WHERE NOT is_deleted
            GROUP BY folder_path"""
        ).fetchall()

        data = {
            "documents": doc_count,
            "chunks": chunk_count,
            "errors": error_count,
            "folders": [
                {
                    "folder_path": row[0],
                    "file_count": row[1],
                    "indexed_count": row[2],
                    "error_count": row[3],
                }
                for row in folder_rows
            ],
        }
        click.echo(json.dumps(data, indent=2))
    else:
        from rag.dashboard import render_dashboard

        render_dashboard(conn, config)


@main.command()
def doctor() -> None:
    """Run health checks on the RAG system."""
    from rag.config import load_config
    from rag.init import check_qdrant_running

    try:
        config = load_config()
        click.echo("Config:    PASS")
    except FileNotFoundError:
        click.echo("Config:    FAIL (no config file found, run 'rag init')")
        return

    # Qdrant
    if check_qdrant_running(config.qdrant.url):
        click.echo("Qdrant:    PASS")
    else:
        click.echo(f"Qdrant:    FAIL (not reachable at {config.qdrant.url})")

    # SQLite
    try:
        from rag.db.connection import get_connection

        conn = get_connection(config.database.path)
        conn.execute("SELECT 1")
        conn.close()
        click.echo("SQLite:    PASS")
    except Exception as e:
        click.echo(f"SQLite:    FAIL ({e})")

    # Folders
    all_ok = True
    for folder in config.folders.paths:
        if not folder.is_dir():
            click.echo(f"Folder:    FAIL ({folder} does not exist)")
            all_ok = False
    if all_ok:
        click.echo(f"Folders:   PASS ({len(config.folders.paths)} configured)")

    # Embedding model cache
    cache_dir = config.embedding.cache_dir
    if cache_dir.is_dir() and any(cache_dir.iterdir()):
        click.echo("Models:    PASS (cache populated)")
    else:
        click.echo("Models:    WARN (model cache empty, first run will download)")


@main.command()
@click.argument("query")
@click.option("--debug", is_flag=True, help="Show timing info.")
@click.option("--top-k", type=int, default=10, help="Number of results.")
def search(query: str, debug: bool, top_k: int) -> None:
    """Search indexed documents."""
    from rag.config import load_config

    config = load_config()
    _db, _runner, engine = _init_components(config)

    result = engine.search(query, top_k=top_k, debug=debug)

    if not result.hits:
        click.echo("No results found.")
        return

    for i, hit in enumerate(result.hits, 1):
        click.echo(f"\n--- Result {i} (score: {hit.score:.4f}) ---")
        click.echo(f"[{hit.citation.label}]")
        text = hit.text[:500] + "..." if len(hit.text) > 500 else hit.text
        click.echo(text)

    if debug and result.debug_info:
        click.echo("\n--- Debug Info ---")
        for key, value in result.debug_info.items():
            click.echo(f"  {key}: {value}")


@main.command(name="mcp-config")
@click.option("--print", "print_config", is_flag=True, help="Print MCP config JSON.")
@click.option(
    "--install",
    type=click.Choice(["claude-desktop", "claude-code", "kiro"]),
    help="Install MCP config for a target.",
)
def mcp_config(print_config: bool, install: str | None) -> None:
    """Print or install MCP server config for Claude Desktop / Claude Code / Kiro."""
    from rag.init import generate_mcp_config, install_mcp_config

    if install is not None:
        ok = install_mcp_config(install)
        if ok:
            click.echo(f"MCP config installed for {install}.")
        else:
            click.echo(f"Failed to install MCP config for {install}.", err=True)
            raise SystemExit(1)
    elif print_config:
        config = generate_mcp_config()
        click.echo(json.dumps(config, indent=2))
    else:
        _show_mcp_help()


def _show_mcp_help() -> None:
    """Show MCP config help with auto-detected tools."""
    from shutil import which

    targets = {
        "claude-code": {"cmd": "claude", "label": "Claude Code"},
        "claude-desktop": {"cmd": None, "label": "Claude Desktop"},
        "kiro": {"cmd": "kiro", "label": "Kiro"},
    }

    click.echo("Install MCP config for your AI tool:\n")

    detected = []
    for target, info in targets.items():
        if info["cmd"] is not None and which(info["cmd"]):
            detected.append(target)
            click.echo(f"  rag mcp-config --install {target:<16} # {info['label']} (detected)")
        elif target == "claude-desktop" and Path(
            "~/Library/Application Support/Claude"
        ).expanduser().exists():
            detected.append(target)
            click.echo(f"  rag mcp-config --install {target:<16} # {info['label']} (detected)")
        else:
            click.echo(f"  rag mcp-config --install {target:<16} # {info['label']}")

    click.echo(f"\n  rag mcp-config --print                   # print raw JSON config")

    if detected:
        click.echo(f"\nDetected: {', '.join(targets[t]['label'] for t in detected)}")


if __name__ == "__main__":
    main()
