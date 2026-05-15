"""
Lightweight XLSX reader for known official workbooks.

This avoids adding an Excel dependency just to extract a few sheets from
trusted external sources such as SPDR / World Gold Council downloads.
It is not intended to be a complete Excel engine; it is a small parser for
simple tabular sheets backed by shared strings.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd

_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
_COL_RE = re.compile(r"[A-Z]+")


@dataclass(frozen=True)
class XlsxSheet:
    name: str
    target: str


def _shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for si in root.findall("a:si", _NS):
        texts = [t.text or "" for t in si.iterfind(".//a:t", _NS)]
        strings.append("".join(texts))
    return strings


def list_sheets(path: str | Path) -> list[XlsxSheet]:
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        sheets: list[XlsxSheet] = []
        for sheet in workbook.find("a:sheets", _NS):
            rid = sheet.attrib.get(f"{{{_NS['r']}}}id")
            target = rel_map.get(rid, "")
            sheets.append(XlsxSheet(name=sheet.attrib.get("name", ""), target=target))
        return sheets


def iter_sheet_rows(path: str | Path, sheet_name: str) -> Iterable[dict[str, str]]:
    """
    Yield each row as a mapping of Excel column letters to string values.
    """
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        sheets = {sheet.name: sheet.target for sheet in list_sheets(path)}
        target = sheets.get(sheet_name)
        if not target:
            raise KeyError(f"Sheet '{sheet_name}' not found. Available: {list(sheets)}")

        shared = _shared_strings(zf)
        sheet_xml = ET.fromstring(zf.read(f"xl/{target}"))
        rows = sheet_xml.find("a:sheetData", _NS)
        if rows is None:
            return

        for row in rows:
            out: dict[str, str] = {}
            for cell in row.findall("a:c", _NS):
                ref = cell.attrib.get("r", "")
                match = _COL_RE.match(ref)
                if not match:
                    continue
                col = match.group(0)
                t = cell.attrib.get("t")
                value_node = cell.find("a:v", _NS)
                value = value_node.text if value_node is not None else ""
                if t == "s" and value != "":
                    value = shared[int(value)]
                out[col] = value
            if out:
                yield out


def sheet_to_frame(
    path: str | Path,
    sheet_name: str,
    *,
    header_row: int = 1,
) -> pd.DataFrame:
    """
    Convert a simple worksheet into a DataFrame using one header row.

    `header_row` is 1-based within the worksheet.
    """
    rows = list(iter_sheet_rows(path, sheet_name))
    if not rows:
        return pd.DataFrame()
    if header_row < 1 or header_row > len(rows):
        raise ValueError(f"header_row out of range: {header_row}")

    header = rows[header_row - 1]
    header_map = {col: name for col, name in header.items() if str(name).strip()}
    records: list[dict[str, str]] = []
    for row in rows[header_row:]:
        record = {
            header_map[col]: value
            for col, value in row.items()
            if col in header_map
        }
        if any(str(v).strip() for v in record.values()):
            records.append(record)
    return pd.DataFrame(records)
