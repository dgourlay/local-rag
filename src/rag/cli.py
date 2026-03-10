from __future__ import annotations

import click


@click.group()
def main() -> None:
    """RAG document retrieval system."""


if __name__ == "__main__":
    main()
