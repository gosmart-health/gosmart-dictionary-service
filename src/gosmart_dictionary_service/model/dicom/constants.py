"""Canonical DICOM terminology URIs, enums, and normalization helpers."""

from __future__ import annotations

import html
import re
from enum import StrEnum
from html.parser import HTMLParser
from typing import Any

# Canonical FHIR-style CodeSystem URIs for DICOM dictionary entities
DICOM_TAG_SYSTEM_URI: str = "http://gosmart.health/terminology/dicom-tag"
DICOM_VR_SYSTEM_URI: str = "http://gosmart.health/terminology/dicom-vr"
DICOM_CIOD_SYSTEM_URI: str = "http://gosmart.health/terminology/dicom-ciod"
DICOM_MODULE_SYSTEM_URI: str = "http://gosmart.health/terminology/dicom-module"
DICOM_MACRO_SYSTEM_URI: str = "http://gosmart.health/terminology/dicom-macro"
DICOM_SOP_SYSTEM_URI: str = "http://gosmart.health/terminology/dicom-sop"
DICOM_CONFIDENTIALITY_SYSTEM_URI: str = "http://gosmart.health/terminology/dicom-confidentiality"
DICOM_REFERENCE_SYSTEM_URI: str = "http://gosmart.health/terminology/dicom-reference"

_TAG_HEX_PATTERN = re.compile(r"^[0-9a-fA-FxX]{8}$")
_HORIZONTAL_WS_PATTERN = re.compile(r"[ \t\r\f\v]+")
_MULTI_NEWLINE_PATTERN = re.compile(r"\n{2,}")

_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "br",
        "li",
        "ul",
        "ol",
        "tr",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "dt",
        "dd",
        "dl",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "pre",
        "section",
    }
)
_INLINE_SEPARATOR_TAGS = frozenset({"td", "th"})
_SKIP_CONTENT_TAGS = frozenset({"script", "style"})


class DicomModuleUsage(StrEnum):
    """DICOM PS3.3 Module / Functional Group Macro usage codes within a CIOD."""

    MANDATORY = "M"
    CONDITIONAL = "C"
    USER_OPTION = "U"


class DicomFuncGroupModuleType(StrEnum):
    """DICOM PS3.3 Functional Group Macro module type classification."""

    MULTI_FRAME = "Multi-frame"
    CURRENT_FRAME = "Current Frame"


class DicomAttributeRequirementType(StrEnum):
    """DICOM PS3.5 / PS3.3 Data Element requirement types within a Module or Macro."""

    TYPE_1 = "1"
    TYPE_1C = "1C"
    TYPE_2 = "2"
    TYPE_2C = "2C"
    TYPE_3 = "3"
    NONE = "None"


class _HTMLTextExtractor(HTMLParser):
    """Stdlib HTML-to-plain-text extractor preserving block boundaries."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        if lower in _SKIP_CONTENT_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if lower in _BLOCK_TAGS:
            self._parts.append("\n")
        elif lower in _INLINE_SEPARATOR_TAGS:
            self._parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower in _SKIP_CONTENT_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if lower in _BLOCK_TAGS:
            self._parts.append("\n")
        elif lower in _INLINE_SEPARATOR_TAGS:
            self._parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = html.unescape("".join(self._parts)).replace("\xa0", " ")
        lines = [_HORIZONTAL_WS_PATTERN.sub(" ", line).strip() for line in raw.splitlines()]
        joined = "\n".join(line for line in lines if line)
        return _MULTI_NEWLINE_PATTERN.sub("\n", joined).strip()


def strip_html_to_text(raw: Any) -> str:
    """Convert an HTML fragment (or plain text) into clean normalized plain text."""
    if raw is None:
        return ""
    text = str(raw)
    if not text.strip():
        return ""
    if "<" not in text and "&" not in text:
        lines = [_HORIZONTAL_WS_PATTERN.sub(" ", line).strip() for line in text.replace("\xa0", " ").splitlines()]
        return "\n".join(line for line in lines if line).strip()
    parser = _HTMLTextExtractor()
    parser.feed(text)
    parser.close()
    return parser.get_text()


def normalize_dicom_tag_hex(raw: str) -> str:
    """Normalize a DICOM tag or 8-char hex ID into canonical uppercase 8-char hex.

    Accepts forms such as ``"00180050"``, ``"300a0782"``, ``"002031xx"``, or
    ``"(0018,0050)"`` and returns ``"00180050"``, ``"300A0782"``, ``"002031XX"``.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("DICOM tag hex identifier must be a non-empty string.")
    cleaned = raw.strip()
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1].replace(",", "").strip()
    elif "," in cleaned:
        cleaned = cleaned.replace(",", "").strip()
    if not _TAG_HEX_PATTERN.match(cleaned):
        raise ValueError(f"Invalid DICOM tag hex identifier: {raw!r}")
    return cleaned.upper()


def normalize_confidentiality_attribute_id(raw: str) -> str:
    """Normalize confidentiality profile attribute IDs.

    Standard 8-char hex / repeating-group hex IDs (e.g. ``"00080050"``,
    ``"50xxxxxx"``) are normalized to uppercase, while descriptive wildcard
    identifiers (e.g. ``"ggggeeee-where-gggg-is-odd"``) are trimmed and preserved.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Confidentiality profile attribute_id must be a non-empty string.")
    cleaned = raw.strip()
    if _TAG_HEX_PATTERN.match(cleaned):
        return cleaned.upper()
    return cleaned


def parse_yn_flag(value: Any) -> bool:
    """Parse a boolean or DICOM ``'Y'`` / ``'N'`` flag string into ``bool``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        norm = value.strip().upper()
        if norm == "Y":
            return True
        if norm == "N":
            return False
    raise ValueError(f"Expected bool or 'Y'/'N' flag, got {value!r}")


__all__ = [
    "DICOM_TAG_SYSTEM_URI",
    "DICOM_VR_SYSTEM_URI",
    "DICOM_CIOD_SYSTEM_URI",
    "DICOM_MODULE_SYSTEM_URI",
    "DICOM_MACRO_SYSTEM_URI",
    "DICOM_SOP_SYSTEM_URI",
    "DICOM_CONFIDENTIALITY_SYSTEM_URI",
    "DICOM_REFERENCE_SYSTEM_URI",
    "DicomModuleUsage",
    "DicomFuncGroupModuleType",
    "DicomAttributeRequirementType",
    "strip_html_to_text",
    "normalize_dicom_tag_hex",
    "normalize_confidentiality_attribute_id",
    "parse_yn_flag",
]
