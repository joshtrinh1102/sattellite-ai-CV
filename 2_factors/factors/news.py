"""Component 2 - news and event ingestion via GDELT.

GDELT 2.0 DOC API is the source (free, no key, reaches back far enough to
cover COVID). It returns headlines and metadata but no article body, so
Component 3 classifies from titles. It does NOT return a per-article tone score - a live probe on 2026-09-22
confirmed the ArtList response carries only domain, language, seendate,
socialimage, sourcecountry, title, url, url_mobile. Sentiment has to be
computed from the titles ourselves; see 2_factors/plan-news.md.

Design notes, query strategy and the sample-size constraint: 2_factors/plan-news.md

Feasibility probe (milestone 1):
    python -m factors.news --probe
"""

import hashlib
import http.client
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# GDELT caps ArtList results per query and gives no cursor, so a busy day for a
# broad query truncates silently. Chunk by day and compare the returned count
# against this to make truncation visible.
MAX_RECORDS = 250

USER_AGENT = "supply-chain-satellite-ai/0.3 (research; contact via repo)"

# Starting port. Aliases matter - most coverage says "Port of LA" or
# "San Pedro Bay", not the full official name.
PORT_QUERIES = {
    "USLAX": '("Port of Los Angeles" OR "Port of Long Beach" OR "San Pedro Bay")',
}


@dataclass
class Article:
    title: str
    url: str
    domain: str
    language: str
    seendate: str
    # Always None from ArtList - GDELT does not return per-article tone.
    # Populated later by our own sentiment pass over `title`.
    tone: float | None = None

    @property
    def date(self):
        """UTC calendar date - the join key against satellite captures."""
        return datetime.strptime(self.seendate, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        ).date()


def _get(params, retries=4, backoff=30.0):
    """One DOC API call, with retries.

    GDELT's rate limit is not a per-second quota that a fixed sleep satisfies -
    sustained use trips a block that returns 429 to *every* request for a
    while. Retrying through that only extends it, which is what `_cooldown`
    above the caller is for.
    """
    url = f"{DOC_API}?{urllib.parse.urlencode(params)}"
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            if not raw.strip():
                return {"articles": []}
            return json.loads(raw)
        # RemoteDisconnected is an HTTPException, not a URLError, so a narrower
        # tuple lets GDELT hanging up mid-run escape the retry loop entirely -
        # which is how a 40-date collection died after one date.
        except (urllib.error.URLError, http.client.HTTPException, OSError,
                json.JSONDecodeError, TimeoutError) as err:
            last_err = err
            if attempt < retries - 1:
                # GDELT rate-limits hard and unpredictably; linear backoff
                # from 15s was what actually got requests through.
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"GDELT request failed after {retries} tries: {last_err}")


def search(query, start, end, max_records=MAX_RECORDS):
    """Fetch articles matching `query` in [start, end).

    Returns (articles, truncated). `truncated` is True when the result hit the
    API cap, meaning the window needs splitting to avoid silent data loss.
    """
    payload = _get({
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max_records,
        "sort": "datedesc",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    })
    articles = [
        Article(
            title=a.get("title", "").strip(),
            url=a.get("url", ""),
            domain=a.get("domain", ""),
            language=a.get("language", ""),
            seendate=a.get("seendate", ""),
            tone=float(a["tone"]) if a.get("tone") not in (None, "") else None,
        )
        for a in payload.get("articles", [])
    ]
    return articles, len(articles) >= max_records


def _article_id(url):
    """Stable id from a normalised URL - the dedupe key.

    Syndication is the dominant duplicate source: the same wire story appears
    under dozens of domains. Normalising away the scheme, `www.`, tracking
    query strings and the trailing slash collapses the ones that really are
    the same URL; near-identical titles across genuinely different domains are
    handled separately, by title, in `collect`.
    """
    u = url.strip().lower()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    u = u.split("?")[0].split("#")[0].rstrip("/")
    return hashlib.sha1(u.encode("utf-8")).hexdigest()[:16]


def _normalise_title(title):
    """Collapse a title to a comparable form for near-duplicate detection."""
    t = title.lower()
    # Outlets append their own name: "... - Reuters", "... | Splash247".
    t = re.split(r"\s+[\-|–—]\s+", t)[0]
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return " ".join(t.split())


def _cooldown(consecutive_failures, base=300.0, cap=1800.0):
    """How long to stand down after `n` consecutive failed dates.

    A 429 from GDELT is a block, not a queue signal: while it is in force every
    request fails, so continuing at the normal cadence just keeps the block
    alive and burns the run in retries. Backing off in minutes - and probing
    once, cheaply, before resuming - is what actually gets the rest of the
    dates collected.
    """
    return min(cap, base * consecutive_failures)


def api_available(query, timeout=45):
    """Single cheap probe: is the API answering us right now?"""
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        _get({
            "query": query, "mode": "ArtList", "format": "json", "maxrecords": 1,
            "startdatetime": (end - timedelta(days=2)).strftime("%Y%m%d%H%M%S"),
            "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        }, retries=1)
        return True
    except (RuntimeError, OSError):
        return False


def search_window(query, start, end, sleep=15.0, max_depth=3):
    """Fetch [start, end), splitting in half whenever the result hits the cap.

    GDELT caps ArtList at 250 with no cursor, so a busy week comes back
    silently clipped - and the busy weeks are the congestion episodes this
    study is about, so what gets dropped is not random. Halving recovers the
    lost articles in 2-4 extra requests where a day-by-day refetch would cost
    seven, which matters because the API rate-limits aggressively enough that
    request count is the binding constraint on a 60-date run.
    """
    articles, truncated = search(query, start, end)
    span_days = (end - start).days
    if not truncated or max_depth <= 0 or span_days <= 1:
        return articles, truncated

    mid = start + timedelta(days=span_days // 2)
    time.sleep(sleep)
    left, lt = search_window(query, start, mid, sleep=sleep, max_depth=max_depth - 1)
    time.sleep(sleep)
    right, rt = search_window(query, mid, end, sleep=sleep, max_depth=max_depth - 1)

    seen, merged = set(), []
    for a in left + right:
        if a.url not in seen:
            seen.add(a.url)
            merged.append(a)
    return merged, (lt or rt)


def cached_days(cache_dir, port_id):
    """Every calendar day already covered by a cached GDELT response.

    Each cache file holds the `window_days` ending on its `capture_date`, so a
    file covers those days whether it was written for a satellite capture (the
    original layout) or for a chunk of a date range (this one).
    """
    days = set()
    for f in cache_dir.glob(f"{port_id}_*d.json"):
        raw = json.loads(f.read_text(encoding="utf-8"))
        end = datetime.strptime(raw["capture_date"], "%Y-%m-%d").date()
        days.update(end - timedelta(days=k) for k in range(raw["window_days"]))
    return days


def collect(port_id, start, end, chunk_days=7, cache_dir=None, sleep=15.0,
            fetch=True):
    """Every port-relevant English headline in [start, end], one row per article.

    Returns (articles, covered) where `covered` is the set of days the cache
    actually spans. Downstream must treat an uncovered day as MISSING, not as a
    day with no news - GDELT rate-limits hard enough that gaps are normal.

    Missing days are fetched in `chunk_days` windows when `fetch` is true. Raw
    responses are cached one file per window, so a run that is cut short by a
    rate-limit block resumes exactly where it stopped.
    """
    from .config import NEWS_RAW

    cache_dir = cache_dir or NEWS_RAW
    cache_dir.mkdir(parents=True, exist_ok=True)
    query = PORT_QUERIES[port_id]
    wanted = {start + timedelta(days=k) for k in range((end - start).days + 1)}

    covered = cached_days(cache_dir, port_id)
    consecutive = 0
    while fetch and (todo := sorted(wanted - covered)):
        chunk_end = min(todo[0] + timedelta(days=chunk_days - 1), end)
        day = datetime(chunk_end.year, chunk_end.month, chunk_end.day, tzinfo=timezone.utc)
        w_end = day + timedelta(days=1)
        w_start = w_end - timedelta(days=chunk_days)
        print(f"  {len(todo)} days left | fetching {w_start.date()} -> {chunk_end}",
              flush=True)
        try:
            articles, truncated = search_window(query, w_start, w_end, sleep=sleep)
        except (RuntimeError, OSError) as err:
            # A 429 is a block, not a queue signal - stand down, probe, resume.
            consecutive += 1
            wait = _cooldown(consecutive)
            print(f"      FAILED ({type(err).__name__}); standing down "
                  f"{wait / 60:.0f} min", flush=True)
            time.sleep(wait)
            while not api_available(query):
                consecutive += 1
                time.sleep(_cooldown(consecutive))
            continue
        cache = cache_dir / f"{port_id}_{chunk_end.isoformat()}_{chunk_days}d.json"
        cache.write_text(json.dumps({
            "capture_date": chunk_end.isoformat(),
            "window_days": chunk_days,
            "truncated": truncated,
            "articles": [vars(a) for a in articles],
        }), encoding="utf-8")
        covered = cached_days(cache_dir, port_id)
        consecutive = 0
        time.sleep(sleep)   # 1s/request triggered HTTP 429 immediately

    rows, seen_ids, seen_titles = [], set(), set()
    for f in sorted(cache_dir.glob(f"{port_id}_*d.json")):
        raw = json.loads(f.read_text(encoding="utf-8"))
        for a in raw["articles"]:
            if not a.get("title") or not a.get("url") or not a.get("seendate"):
                continue
            if not a.get("language", "").lower().startswith("eng"):
                continue
            day = datetime.strptime(a["seendate"][:8], "%Y%m%d").date()
            if day not in wanted:
                continue
            # Overlapping cache windows share articles; each counts once, on the
            # day GDELT first saw it.
            aid, norm = _article_id(a["url"]), _normalise_title(a["title"])
            if aid in seen_ids or norm in seen_titles or len(norm) < 12:
                continue
            seen_ids.add(aid)
            seen_titles.add(norm)
            rows.append({
                "article_id": aid,
                "date": day.isoformat(),
                "port_id": port_id,
                "title": a["title"],
                "url": a["url"],
                "domain": a.get("domain", ""),
                "truncated": bool(raw.get("truncated")),
            })

    covered &= wanted
    print(f"  news: {len(rows)} unique English headlines, "
          f"{len(covered)}/{len(wanted)} days covered")
    return rows, covered


def probe(port_id="USLAX", days=7, end=None):
    """Milestone 1: is GDELT coverage dense enough to build on?

    Target from the plan is >=5 relevant articles/day. Prints a per-day count
    and a sample of titles so the signal can be judged by eye - titles are the
    only text Component 3 will get, so they have to carry the category.
    """
    query = PORT_QUERIES[port_id]
    end = end or datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start = end - timedelta(days=days)

    print(f"GDELT probe | {port_id} | {start.date()} -> {end.date()}")
    print(f"query: {query}\n")

    per_day, all_articles = {}, []
    for offset in range(days):
        day_start = start + timedelta(days=offset)
        day_end = day_start + timedelta(days=1)
        articles, truncated = search(query, day_start, day_end)
        per_day[day_start.date()] = len(articles)
        all_articles.extend(articles)
        flag = "  <-- TRUNCATED, split this window" if truncated else ""
        print(f"  {day_start.date()}  {len(articles):3d} articles{flag}")
        time.sleep(6.0)   # 1s/request triggered HTTP 429 immediately

    counts = list(per_day.values())
    total = sum(counts)
    mean = total / len(counts) if counts else 0
    english = sum(1 for a in all_articles if a.language.lower().startswith("eng"))
    toned = [a.tone for a in all_articles if a.tone is not None]
    unique_titles = len({a.title.lower() for a in all_articles if a.title})

    print(f"\n  total {total} | mean {mean:.1f}/day | "
          f"english {english}/{total if total else 1}")
    print(f"  unique titles {unique_titles}/{total} "
          f"(the gap is syndication - dedupe will remove it)")
    if toned:
        print(f"  tone present on {len(toned)}/{total}, "
              f"mean {sum(toned) / len(toned):+.2f}")
    else:
        print("  tone NOT returned by this endpoint - check the API response shape")

    verdict = "PASS" if mean >= 5 else "THIN"
    print(f"\n  verdict: {verdict} (plan target is >=5 relevant articles/day)")
    if verdict == "THIN":
        print("  -> broaden the query, widen to more ports, or reconsider the source")

    print("\n  sample titles:")
    for a in all_articles[:12]:
        tone = f"{a.tone:+.1f}" if a.tone is not None else "  n/a"
        print(f"    [{tone}] {a.title[:96]}")

    return per_day, all_articles


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="GDELT ingestion for Component 2.")
    p.add_argument("--probe", action="store_true",
                   help="run the feasibility probe (milestone 1)")
    p.add_argument("--port", default="USLAX", choices=sorted(PORT_QUERIES))
    p.add_argument("--days", type=int, default=7)
    args = p.parse_args(argv)

    if args.probe:
        probe(port_id=args.port, days=args.days)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
