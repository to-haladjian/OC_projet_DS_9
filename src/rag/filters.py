"""Call 1 of the RAG chain: extract structured filters and turn them into a predicate.

The LLM is asked (via :data:`FILTER_EXTRACTION_PROMPT`) for a small JSON object describing
the city / department / date window mentioned in the question. ``extract_filters`` parses
that defensively, and ``build_metadata_filter`` compiles it into a callable used by
``FAISS.similarity_search(filter=...)`` to pre-filter retrieval on Document metadata.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Callable

from src.rag.prompts import FILTER_EXTRACTION_PROMPT

# Keys we accept from the model; anything else is ignored.
_FILTER_KEYS = ("city", "department", "date_from", "date_to")

_FENCE_OPEN = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\s*```$")
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _message_text(response: object) -> str:
    """Extract plain text from an LLM response (string, AIMessage, or content list)."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # some providers return a list of content parts
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _parse_filter_json(text: str) -> dict:
    """Parse the model's reply into a dict, tolerating fences and surrounding prose."""
    text = text.strip()
    text = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", text))
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(text)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


def extract_filters(question: str, llm, today: date) -> dict:
    """Ask the LLM for structured filters; return a clean dict (``{}`` on failure).

    Only the known keys with a non-empty value are kept, so a fully-null answer (no
    constraints in the question) yields ``{}`` and disables pre-filtering downstream.
    """
    prompt = FILTER_EXTRACTION_PROMPT.format(today=today.isoformat())
    response = llm.invoke([("system", prompt), ("human", question)])
    data = _parse_filter_json(_message_text(response))

    filters: dict = {}
    for key in _FILTER_KEYS:
        value = data.get(key)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            filters[key] = value
    return filters


def _parse_date(value: object) -> date | None:
    """Parse an ISO date/datetime string to a ``date`` (None if missing/unparseable)."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nat":
        return None
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


def build_metadata_filter(filters: dict) -> Callable[[dict], bool] | None:
    """Compile extracted filters into a metadata predicate (None if there is nothing).

    Date handling is intentionally lenient: an event whose dates cannot be parsed is
    kept rather than dropped, to avoid over-filtering on imperfect metadata.
    """
    if not filters:
        return None

    city = filters.get("city")
    department = filters.get("department")
    date_from = _parse_date(filters.get("date_from"))
    date_to = _parse_date(filters.get("date_to"))

    def predicate(metadata: dict) -> bool:
        if department and str(metadata.get("department", "")) != department:
            return False
        if city and str(metadata.get("city", "")).casefold() != city.casefold():
            return False
        if date_from or date_to:
            start = _parse_date(metadata.get("date_start"))
            end = _parse_date(metadata.get("date_end")) or start
            start = start or end
            if start is not None and end is not None:
                if date_to and start > date_to:
                    return False
                if date_from and end < date_from:
                    return False
        return True

    return predicate
