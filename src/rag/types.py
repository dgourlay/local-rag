from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel

# --- UUID namespace for deterministic chunk IDs ---
NAMESPACE_RAG = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

# --- Constrained value types ---
ProcessStatus = Literal["pending", "processing", "done", "error", "poison"]
SummaryLevel = Literal["l1", "l2", "l3"]


class RecordType(StrEnum):
    CHUNK = "chunk"
    SECTION_SUMMARY = "section_summary"
    DOCUMENT_SUMMARY = "document_summary"


class FileType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"


class ProcessingOutcome(StrEnum):
    INDEXED = "indexed"
    UNCHANGED = "unchanged"
    DUPLICATE = "duplicate"
    DELETED = "deleted"
    ERROR = "error"


# --- Pipeline boundary models ---


class ParsedSection(BaseModel):
    heading: str | None
    order: int
    text: str
    page_start: int | None = None
    page_end: int | None = None


class ParsedDocument(BaseModel):
    doc_id: str
    title: str | None
    file_type: FileType
    sections: list[ParsedSection]
    ocr_required: bool = False
    ocr_confidence: float | None = None
    raw_content_hash: str


class ClassificationResult(BaseModel):
    file_type: FileType
    likely_scanned: bool
    ocr_enabled: bool
    folder_context: str
    complexity_estimate: Literal["low", "medium", "high"]


class NormalizedDocument(BaseModel):
    doc_id: str
    title: str | None
    file_type: FileType
    sections: list[ParsedSection]
    normalized_content_hash: str
    raw_content_hash: str


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    section_id: str | None = None
    chunk_order: int
    text: str
    text_normalized: str
    page_start: int | None = None
    page_end: int | None = None
    section_heading: str | None = None
    citation_label: str | None = None
    token_count: int


class EmbeddedChunk(BaseModel):
    chunk: Chunk
    vector: list[float]


class QdrantPayloadModel(BaseModel):
    record_type: RecordType
    summary_level: SummaryLevel | None = None
    doc_id: str
    section_id: str | None = None
    chunk_id: str | None = None
    title: str
    file_path: str
    folder_path: str
    folder_ancestors: list[str]
    file_type: FileType
    modified_at: str
    page_start: int | None = None
    page_end: int | None = None
    section_heading: str | None = None
    chunk_order: int | None = None
    doc_type_guess: str | None = None
    ocr_confidence: float | None = None
    token_count: int | None = None
    citation_label: str | None = None
    text: str


class VectorPoint(BaseModel):
    point_id: str
    vector: list[float]
    payload: QdrantPayloadModel


class FileEvent(BaseModel):
    file_path: str
    content_hash: str
    file_type: FileType
    event_type: Literal["created", "modified", "deleted"]
    modified_at: str


class SearchFilters(BaseModel):
    folder_filter: str | None = None
    date_filter: str | None = None
    file_type: FileType | None = None


class SearchHit(BaseModel):
    point_id: str
    score: float
    record_type: RecordType
    doc_id: str
    text: str
    payload: dict[str, Any]


class Citation(BaseModel):
    title: str
    path: str
    section: str | None = None
    pages: str | None = None
    modified: str
    label: str


class CitedEvidence(BaseModel):
    text: str
    citation: Citation
    score: float
    record_type: str
    doc_id: str


class RetrievalResult(BaseModel):
    hits: list[CitedEvidence]
    query_classification: str | None = None
    debug_info: dict[str, Any] | None = None


# --- Qdrant payload read-back ---


class QdrantPayloadReadBack(TypedDict):
    record_type: str
    summary_level: str | None
    doc_id: str
    section_id: str | None
    chunk_id: str | None
    title: str
    file_path: str
    folder_path: str
    folder_ancestors: list[str]
    file_type: str
    modified_at: str
    text: str


# --- Frozen dataclasses for stage-internal value objects ---


@dataclass(frozen=True, slots=True)
class ChunkWindow:
    center: Chunk
    before: list[Chunk] = field(default_factory=list)
    after: list[Chunk] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RRFCandidate:
    point_id: str
    rrf_score: float
    source_ranks: dict[str, int] = field(default_factory=dict)


# --- DB row models (map to SQLite tables) ---


class SyncStateRow(BaseModel):
    id: str
    file_path: str
    file_name: str
    folder_path: str
    folder_ancestors: list[str]
    file_type: str
    size_bytes: int | None = None
    modified_at: str
    content_hash: str
    synced_at: str | None = None
    process_status: ProcessStatus = "pending"
    error_message: str | None = None
    retry_count: int = 0
    is_deleted: int = 0
    created_at: str | None = None


class DocumentRow(BaseModel):
    doc_id: str
    file_path: str
    folder_path: str
    folder_ancestors: list[str]
    title: str | None = None
    file_type: str
    modified_at: str
    indexed_at: str | None = None
    parser_version: str | None = None
    raw_content_hash: str
    normalized_content_hash: str | None = None
    duplicate_of_doc_id: str | None = None
    ocr_required: int = 0
    ocr_confidence: float | None = None
    doc_type_guess: str | None = None
    key_topics: list[str] | None = None
    summary_l1: str | None = None
    summary_l2: str | None = None
    summary_l3: str | None = None
    summary_content_hash: str | None = None
    embedding_model_version: str | None = None
    chunker_version: str | None = None


class SectionRow(BaseModel):
    section_id: str
    doc_id: str
    section_heading: str | None = None
    section_order: int
    page_start: int | None = None
    page_end: int | None = None
    section_summary: str | None = None
    section_summary_l2: str | None = None
    embedding_model_version: str | None = None


class ChunkRow(BaseModel):
    chunk_id: str
    doc_id: str
    section_id: str | None = None
    chunk_order: int
    chunk_text: str
    chunk_text_normalized: str
    page_start: int | None = None
    page_end: int | None = None
    section_heading: str | None = None
    citation_label: str | None = None
    token_count: int | None = None
    embedding_model_version: str | None = None


class ProcessingLogEntry(BaseModel):
    doc_id: str | None = None
    file_path: str | None = None
    stage: str
    status: str
    duration_ms: int | None = None
    details: str | None = None
    created_at: str | None = None


# --- MCP tool input/output models ---


class SearchDocumentsInput(BaseModel):
    query: str
    folder_filter: str | None = None
    date_filter: str | None = None
    top_k: int = 10
    debug: bool = False
    format: Literal["text", "json"] = "text"


class QuickSearchInput(BaseModel):
    query: str
    folder_filter: str | None = None
    top_k: int = 5


class SearchDocumentsOutput(BaseModel):
    results: list[CitedEvidence]
    query_classification: str | None = None
    debug_info: dict[str, Any] | None = None


class GetDocumentContextInput(BaseModel):
    doc_id: str | None = None
    chunk_id: str | None = None
    window: int = 1


class DocumentContextOutput(BaseModel):
    doc_id: str
    title: str | None = None
    summary: str | None = None
    sections: list[SectionRow] | None = None
    chunks: list[ChunkRow] | None = None


class ListRecentDocumentsInput(BaseModel):
    folder_filter: str | None = None
    limit: int = 20


class RecentDocumentEntry(BaseModel):
    doc_id: str
    title: str | None = None
    file_path: str
    file_type: str
    modified_at: str
    indexed_at: str | None = None
    folder_path: str


class ListRecentDocumentsOutput(BaseModel):
    documents: list[RecentDocumentEntry]


class FolderStatusEntry(BaseModel):
    folder_path: str
    file_count: int
    indexed_count: int
    error_count: int


class SyncStatusOutput(BaseModel):
    total_files: int
    indexed_count: int
    pending_count: int
    error_count: int
    last_sync_time: str | None = None
    folders: list[FolderStatusEntry]


# Rebuild forward refs
VectorPoint.model_rebuild()
RetrievalResult.model_rebuild()
SyncStatusOutput.model_rebuild()
