"""
URL validator — drops dead links from LLM-generated search results.

Conservative rules: only drop on 404, 410, DNS failure, connection error, or timeout.
Keep 401/403/429/5xx/redirects — those usually mean the article exists but the
publisher blocked our datacenter IP or rate-limited us.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

import requests

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

DROP_STATUSES = {404, 410}


def _check_url(url: str, timeout: float) -> bool:
    """Return True if URL is OK to keep, False to drop."""
    if not url or not isinstance(url, str):
        return False

    headers = {"User-Agent": BROWSER_UA, "Accept": "*/*"}

    try:
        resp = requests.head(url, headers=headers, allow_redirects=True, timeout=timeout)
        status = resp.status_code

        # Some sites reject HEAD — retry with GET
        if status in (405, 501):
            resp = requests.get(url, headers=headers, allow_redirects=True,
                                timeout=timeout, stream=True)
            status = resp.status_code
            resp.close()

        return status not in DROP_STATUSES

    except (requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.TooManyRedirects):
        return False
    except requests.exceptions.RequestException:
        # Unknown request error — keep it (fail open)
        return True
    except Exception:
        return True


def filter_valid_urls(
    results: List[Dict],
    url_key: str = "url",
    max_workers: int = 10,
    timeout: float = 3.0,
) -> List[Dict]:
    """
    Filter a list of result dicts, dropping entries whose URL hard-404s.

    Preserves input order. Runs checks concurrently.
    """
    if not results:
        return results

    keep_flags: Dict[int, bool] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(_check_url, r.get(url_key, ""), timeout): i
            for i, r in enumerate(results)
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                keep_flags[idx] = fut.result()
            except Exception:
                keep_flags[idx] = True  # fail open

    kept = [r for i, r in enumerate(results) if keep_flags.get(i, True)]
    dropped = len(results) - len(kept)
    if dropped:
        print(f"[url_validator] Dropped {dropped}/{len(results)} dead links")
    return kept
