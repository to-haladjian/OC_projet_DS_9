"""Pydantic request/response models for the API (also drive the Swagger schema)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    """A natural-language question about Hauts-de-Seine (92) events."""

    question: str = Field(
        ...,
        min_length=1,
        examples=["Quels concerts de jazz à Nanterre ce week-end ?"],
    )

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty")
        return value


class AskResponse(BaseModel):
    """The grounded answer plus the filters extracted and the events cited."""

    answer: str
    filters: dict[str, str]
    sources: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    documents: int


class MetadataResponse(BaseModel):
    """Aggregate stats about the indexed corpus (no LLM calls)."""

    events: int = Field(..., description="Distinct events indexed (deduplicated by id).")
    cities: int = Field(..., description="Number of distinct cities.")
    departments: dict[str, int] = Field(
        ..., description="Event count per department code (Hauts-de-Seine = 92)."
    )
    date_range: dict[str, str | None] = Field(
        ...,
        description="Earliest and latest event dates, as {'from': ..., 'to': ...}.",
    )


class RebuildResponse(BaseModel):
    status: str
