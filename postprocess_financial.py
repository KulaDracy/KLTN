# postprocess_financial_v5.py
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict

VN_CURRENCY_HINTS = ["dong", "đồng", "vnd", "viet nam"]

SECTION_PATTERNS = {
    "ASSETS_SHORT": r"TÀI\s*SẢN\s*NGẮN\s*HẠN",
    "ASSETS_LONG": r"TÀI\s*SẢN\s*DÀI\s*HẠN",
    "LIABILITIES": r"NỢ\s*PHẢI\s*TRẢ",
    "EQUITY": r"VỐN\s*CHỦ\s*SỞ\s*HỮU",
}

LINE_CODE_REGEX = re.compile(r"^\s*(\d{2,3})\s+(.+?)\s{2,}")

NUMBER_REGEX = re.compile(r"[\d][\d\.,;]*")

DATE_REGEX = re.compile(r"(\d{2}/\d{2}/\d{4})")


# -------------------------
# Utils
# -------------------------

def normalize_number(text: str) -> Optional[int]:
    """
    Normalize Vietnamese financial numbers safely.
    """
    raw = re.sub(r"[^\d]", "", text)
    if len(raw) < 4:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def extract_all_numbers(text: str) -> List[int]:
    nums = []
    for m in NUMBER_REGEX.findall(text):
        val = normalize_number(m)
        if val:
            nums.append(val)
    return nums


# -------------------------
# Main Processor
# -------------------------

class FinancialPostProcessor:
    def __init__(self):
        pass

    # ===== META =====
    def extract_meta(self, text: str) -> Dict[str, Any]:
        meta = {
            "company_name": None,
            "report_type": "BALANCE_SHEET",
            "dates": {},
            "currency": None,
        }

        # Company name
        for line in text.splitlines():
            if "CONG TY" in line.upper():
                meta["company_name"] = (
                    line.replace("CONG TY", "CÔNG TY")
                        .replace("C PHAN", "CỔ PHẦN")
                        .strip()
                )
                break

        # Dates
        dates = DATE_REGEX.findall(text)
        if len(dates) >= 1:
            meta["dates"]["current"] = dates[0]
        if len(dates) >= 2:
            meta["dates"]["previous"] = dates[1]

        # Currency
        lower = text.lower()
        if any(h in lower for h in VN_CURRENCY_HINTS):
            meta["currency"] = "VND"

        return meta

    # ===== TABLE PARSING =====
    def parse_rows_from_lines(self, lines: List[Dict]) -> List[Dict]:
        """
        Fallback layout-based parsing (no table bbox)
        """
        rows = []

        for line in lines:
            txt = line["text"]
            code_match = LINE_CODE_REGEX.match(txt)
            if not code_match:
                continue

            code = code_match.group(1)
            name = code_match.group(2).strip()
            numbers = extract_all_numbers(txt)

            if len(numbers) >= 2:
                rows.append({
                    "code": code,
                    "name": name,
                    "values": numbers[:2]
                })

        return rows

    def parse_table_with_bbox(self, table: Dict) -> List[Dict]:
        """
        True table-aware parsing
        """
        rows = []

        for row in table.get("rows", []):
            if len(row) < 3:
                continue

            code = row[0].get("text", "").strip()
            name = row[1].get("text", "").strip()
            v1 = normalize_number(row[2].get("text", ""))
            v2 = normalize_number(row[3].get("text", "")) if len(row) > 3 else None

            if code.isdigit() and name and v1:
                rows.append({
                    "code": code,
                    "name": name,
                    "values": [v1, v2]
                })

        return rows

    # ===== BUILD BALANCE SHEET =====
    def build_balance_sheet(self, rows: List[Dict]) -> List[Dict]:
        sections = defaultdict(list)
        current_section = None

        for r in rows:
            for key, pat in SECTION_PATTERNS.items():
                if re.search(pat, r["name"].upper()):
                    current_section = key
                    break

            if current_section:
                sections[current_section].append(r)

        return [
            {
                "section": k,
                "items": v
            }
            for k, v in sections.items()
        ]

    # ===== ENTRY POINT =====
    def process_page(self, page_ocr: Dict) -> Dict[str, Any]:
        text = page_ocr.get("content", "")
        blocks = page_ocr.get("text_blocks", [])
        tables = page_ocr.get("tables", [])

        meta = self.extract_meta(text)

        rows = []
        if tables:
            for t in tables:
                rows.extend(self.parse_table_with_bbox(t))
        else:
            rows = self.parse_rows_from_lines(blocks)

        balance_sheet = self.build_balance_sheet(rows)

        return {
            **meta,
            "sections": balance_sheet
        }
