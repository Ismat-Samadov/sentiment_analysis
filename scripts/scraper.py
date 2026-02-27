"""
Scraper for supertoys.az/oyuncaqlar-105
Fetches all pages async and saves product data to CSV.

Usage:
    pip install aiohttp
    python scripts/scraper.py
"""

import asyncio
import csv
import json
import re
import sys
import time
from pathlib import Path

import aiohttp

BASE_URL = "https://www.supertoys.az/oyuncaqlar-105"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "products.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "az,en;q=0.9",
}

CONCURRENCY = 10          # parallel requests
REQUEST_DELAY = 0.3       # seconds between batches

CSV_FIELDS = [
    "id", "name", "code", "supplier_code",
    "sale_price", "total_base_price", "total_sale_price", "vat",
    "currency", "quantity",
    "brand", "category", "category_id", "category_path",
    "model", "url", "image",
    "variant1", "variant2",
    "subproduct_code", "subproduct_id", "personalization_id",
]

PRODUCT_RE = re.compile(
    r"PRODUCT_DATA\.push\(JSON\.parse\('(.+?)'\)\)",
    re.DOTALL,
)

TOTAL_PAGES_RE = re.compile(r'pg=(\d+)[^"]*"[^>]*>\s*(\d+)\s*</a>', re.IGNORECASE)


def _unescape_js_string(s: str) -> str:
    """Unescape a JS single-quoted string literal.

    Processes escape sequences left-to-right so that \\\\ → \\ and \\" → "
    are handled correctly without double-substitution.  \\uXXXX and other
    JSON escape sequences are preserved for json.loads to handle.
    """
    result: list[str] = []
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt == "\\":
                result.append("\\")
                i += 2
            elif nxt == "'":
                result.append("'")
                i += 2
            elif nxt == '"':
                result.append('"')
                i += 2
            else:
                # Keep other escape sequences (\uXXXX, \n, \/, …) intact
                # so json.loads can interpret them.
                result.append("\\")
                result.append(nxt)
                i += 2
        else:
            result.append(s[i])
            i += 1
    return "".join(result)


def parse_products(html: str) -> list[dict]:
    products = []
    for m in PRODUCT_RE.finditer(html):
        raw = _unescape_js_string(m.group(1))
        try:
            products.append(json.loads(raw))
        except json.JSONDecodeError:
            pass
    return products


def get_total_pages(html: str) -> int:
    """Extract max page number from pagination links."""
    numbers = [int(n) for n in re.findall(r'[?&]pg=(\d+)', html)]
    return max(numbers) if numbers else 1


async def fetch_page(session: aiohttp.ClientSession, page: int) -> str:
    url = f"{BASE_URL}?pg={page}"
    async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30)) as resp:
        resp.raise_for_status()
        return await resp.text()


async def scrape_all() -> list[dict]:
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        # --- page 1: discover total pages ---
        print("Fetching page 1 to discover total pages...")
        html1 = await fetch_page(session, 1)
        total_pages = get_total_pages(html1)
        print(f"Total pages: {total_pages}")

        all_products: list[dict] = parse_products(html1)
        print(f"  Page 1: {len(all_products)} products")

        # --- remaining pages in concurrent batches ---
        remaining = list(range(2, total_pages + 1))
        sem = asyncio.Semaphore(CONCURRENCY)

        async def fetch_with_sem(page: int) -> tuple[int, str]:
            async with sem:
                html = await fetch_page(session, page)
                return page, html

        total_fetched = len(all_products)
        for batch_start in range(0, len(remaining), CONCURRENCY * 5):
            batch = remaining[batch_start: batch_start + CONCURRENCY * 5]
            tasks = [fetch_with_sem(p) for p in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    print(f"  Error: {result}", file=sys.stderr)
                    continue
                page, html = result
                products = parse_products(html)
                all_products.extend(products)
                total_fetched += len(products)
                print(f"  Page {page}: {len(products)} products  (running total: {total_fetched})")

            if batch_start + CONCURRENCY * 5 < len(remaining):
                await asyncio.sleep(REQUEST_DELAY)

    return all_products


def save_csv(products: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(products)
    print(f"\nSaved {len(products)} products -> {path}")


def main() -> None:
    t0 = time.perf_counter()
    products = asyncio.run(scrape_all())
    elapsed = time.perf_counter() - t0

    if not products:
        print("No products found. Check PRODUCT_RE pattern or site structure.")
        sys.exit(1)

    # de-duplicate by product id
    seen: set[str] = set()
    unique = []
    for p in products:
        pid = str(p.get("id", ""))
        if pid not in seen:
            seen.add(pid)
            unique.append(p)

    print(f"\nTotal scraped : {len(products)}")
    print(f"After dedup   : {len(unique)}")
    print(f"Elapsed       : {elapsed:.1f}s")

    save_csv(unique, OUTPUT_FILE)


if __name__ == "__main__":
    main()
