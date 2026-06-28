"""Clean and structure the raw OpenAgenda events into a RAG-ready dataset.

The raw export (``data/openagenda_92_events.csv``) is noisy: descriptions are raw HTML,
dates are unparsed strings, postal codes come back as floats, departments are free-text
names with many spelling variants, and there are duplicate events. This module turns that
into a tidy, structured DataFrame (the columns of :data:`src.config.SCHEMA_COLUMNS`),
which is the single input to FAISS indexing.

Design choice: we keep only fields that map *reliably* from the source. We deliberately do
NOT invent ``category``/``price_type`` from the noisy free-text ``keywords_fr`` /
``conditions_fr``; those are kept as raw context columns instead.
"""

from __future__ import annotations

import ast
import html
import re
import unicodedata
import warnings

import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

from src.config import SCHEMA_COLUMNS, TARGET_DEPARTMENT_CODES

# Many descriptions are plain one-liners (sometimes a bare URL); BeautifulSoup warns that
# they "resemble a locator" rather than markup. That's expected here, so silence it.
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

_WHITESPACE = re.compile(r"\s+")

# Normalised department name -> INSEE code, used as a fallback when the postal code is
# missing. Keys are accent/punctuation-stripped lowercase (see ``_normalize_name``), so
# all the spelling variants seen in the data ("Seine-St-Denis", "Val-D'Oise", ...) map.
_NAME_TO_CODE = {
    "paris": "75",
    "seineetmarne": "77",
    "yvelines": "78",
    "essonne": "91",
    "hautsdeseine": "92",
    "seinesaintdenis": "93",
    "seinestdenis": "93",
    "valdemarne": "94",
    "valdoise": "95",
}


def _is_missing(value: object) -> bool:
    """True for NaN / None / pd.NA scalars (robust to non-scalar inputs)."""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _clean_text(value: object) -> object:
    """Collapse whitespace and strip; return ``pd.NA`` for empty/missing values."""
    if _is_missing(value):
        return pd.NA
    text = _WHITESPACE.sub(" ", str(value)).strip()
    return text if text else pd.NA


def strip_html(value: object) -> str:
    """Strip HTML tags, unescape entities and collapse whitespace.

    Returns an empty string for missing values (callers decide whether "" means drop).
    """
    if _is_missing(value):
        return ""
    text = BeautifulSoup(str(value), "html.parser").get_text(separator=" ")
    text = html.unescape(text)
    return _WHITESPACE.sub(" ", text).strip()


def normalize_postalcode(value: object) -> object:
    """Turn a postal code (often read as ``75014.0``) into a 5-char string."""
    if _is_missing(value):
        return pd.NA
    try:
        code = str(int(float(value)))
    except (TypeError, ValueError):
        code = str(value).strip()
    code = code.zfill(5)
    return code if code else pd.NA


def _normalize_name(value: object) -> str:
    """Lowercase, drop accents and keep only alphanumerics (for name matching)."""
    if _is_missing(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", text.lower())


def resolve_department(postalcode: object, name: object) -> object:
    """Resolve a 2-digit department code from the postal code, falling back to name.

    A postal code is authoritative (its first two digits are the department); only when
    it is missing do we try to recognise the free-text department name. The returned code
    is later filtered against :data:`TARGET_DEPARTMENT_CODES`, so an out-of-scope postal
    code (e.g. Paris' ``75...``) is still resolved here and simply dropped downstream.
    """
    if not _is_missing(postalcode):
        return str(postalcode)[:2]
    return _NAME_TO_CODE.get(_normalize_name(name), pd.NA)


def normalize_keywords(value: object) -> object:
    """Flatten the stringified list ``"['Jazz', 'Concert']"`` into ``"Jazz, Concert"``."""
    if _is_missing(value):
        return pd.NA
    raw = str(value).strip()
    if not raw:
        return pd.NA
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw
    if isinstance(parsed, (list, tuple)):
        items = [str(x).strip() for x in parsed if str(x).strip()]
        return ", ".join(items) if items else pd.NA
    return raw


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw events DataFrame into the structured :data:`SCHEMA_COLUMNS`.

    Steps: strip HTML, drop text-less events, parse dates, normalise postal codes,
    resolve department codes, keep only Hauts-de-Seine (92) events, and deduplicate on the
    event id (keeping the most recently updated record).
    """
    df = df.copy()

    # --- Text fields (HTML-stripped) ---
    description = df["longdescription_fr"].map(strip_html)
    summary = df["description_fr"].map(strip_html)
    # Fall back to the short description when the long one is empty.
    description = description.where(description.str.len() > 0, summary)
    df["description"] = description
    df["summary"] = summary.where(summary.str.len() > 0, pd.NA)
    df["conditions"] = df["conditions_fr"].map(strip_html)
    df["conditions"] = df["conditions"].where(df["conditions"].str.len() > 0, pd.NA)
    df["title"] = df["title_fr"].map(_clean_text)

    # Drop events with no usable text at all (nothing to embed nor display).
    has_text = (df["description"].str.len() > 0) | df["title"].notna()
    df = df[has_text]

    # --- Dates (tz-aware datetimes; NaT for missing/unparseable) ---
    df["date_start"] = pd.to_datetime(df["firstdate_begin"], utc=True, errors="coerce")
    df["date_end"] = pd.to_datetime(df["lastdate_end"], utc=True, errors="coerce")
    df["daterange"] = df["daterange_fr"].map(_clean_text)
    df["updatedat"] = pd.to_datetime(df["updatedat"], utc=True, errors="coerce")

    # --- Location ---
    df["postalcode"] = df["location_postalcode"].map(normalize_postalcode)
    df["department"] = [
        resolve_department(pc, name)
        for pc, name in zip(df["postalcode"], df["location_department"])
    ]
    df["city"] = df["location_city"].map(_clean_text)
    df["venue"] = df["location_name"].map(_clean_text)
    df["address"] = df["location_address"].map(_clean_text)
    df["coordinates"] = df["location_coordinates"].map(_clean_text)

    # --- Misc ---
    df["id"] = df["uid"]
    df["url"] = df["canonicalurl"].map(_clean_text)
    df["keywords"] = df["keywords_fr"].map(normalize_keywords)

    # Keep only Hauts-de-Seine events, then drop duplicate ids (newest update wins).
    df = df[df["department"].isin(TARGET_DEPARTMENT_CODES)]
    df = df.sort_values("updatedat", na_position="first").drop_duplicates(
        subset="id", keep="last"
    )

    return df[SCHEMA_COLUMNS].reset_index(drop=True)


def null_report(df: pd.DataFrame) -> pd.Series:
    """Per-column count of missing values (for the CLI's data-quality summary)."""
    return df.isna().sum()


def save_clean(df: pd.DataFrame, path) -> None:
    """Write the cleaned DataFrame to CSV (datetimes serialised as ISO-8601)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
