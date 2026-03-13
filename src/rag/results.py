from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, Discriminator, Tag

if TYPE_CHECKING:
    from rag.types import ParsedDocument


class ParseSuccess(BaseModel):
    status: Literal["success"] = "success"
    document: ParsedDocument


class ParseError(BaseModel):
    status: Literal["error"] = "error"
    error: str
    file_path: str


ParseResult = Annotated[
    Annotated[ParseSuccess, Tag("success")] | Annotated[ParseError, Tag("error")],
    Discriminator("status"),
]


class SummarySuccess(BaseModel):
    status: Literal["success"] = "success"
    summary_8w: str
    summary_16w: str
    summary_32w: str
    summary_64w: str
    summary_128w: str
    key_topics: list[str]
    doc_type_guess: str | None = None


class SummaryError(BaseModel):
    status: Literal["error"] = "error"
    error: str


SummaryResult = Annotated[
    Annotated[SummarySuccess, Tag("success")] | Annotated[SummaryError, Tag("error")],
    Discriminator("status"),
]


class SectionSummarySuccess(BaseModel):
    status: Literal["success"] = "success"
    section_summary_8w: str
    section_summary_32w: str
    section_summary_128w: str


class SectionSummaryError(BaseModel):
    status: Literal["error"] = "error"
    error: str


SectionSummaryResult = Annotated[
    Annotated[SectionSummarySuccess, Tag("success")]
    | Annotated[SectionSummaryError, Tag("error")],
    Discriminator("status"),
]


class CombinedSectionSummary(BaseModel):
    heading: str | None = None
    section_summary_8w: str
    section_summary_32w: str
    section_summary_128w: str


class CombinedSummarySuccess(BaseModel):
    status: Literal["success"] = "success"
    summary_8w: str
    summary_16w: str
    summary_32w: str
    summary_64w: str
    summary_128w: str
    key_topics: list[str]
    doc_type_guess: str | None = None
    sections: list[CombinedSectionSummary] = []


class CombinedSummaryError(BaseModel):
    status: Literal["error"] = "error"
    error: str


CombinedSummaryResult = Annotated[
    Annotated[CombinedSummarySuccess, Tag("success")]
    | Annotated[CombinedSummaryError, Tag("error")],
    Discriminator("status"),
]


class EmbedSuccess(BaseModel):
    status: Literal["success"] = "success"
    vectors: list[list[float]]


class EmbedError(BaseModel):
    status: Literal["error"] = "error"
    error: str


EmbedResult = Annotated[
    Annotated[EmbedSuccess, Tag("success")] | Annotated[EmbedError, Tag("error")],
    Discriminator("status"),
]


class IndexSuccess(BaseModel):
    status: Literal["success"] = "success"
    points_upserted: int
    points_deleted: int


class IndexingError(BaseModel):
    status: Literal["error"] = "error"
    error: str


IndexResult = Annotated[
    Annotated[IndexSuccess, Tag("success")] | Annotated[IndexingError, Tag("error")],
    Discriminator("status"),
]
