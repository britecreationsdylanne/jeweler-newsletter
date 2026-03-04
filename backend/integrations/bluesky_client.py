"""
Bluesky Search Client

Uses the public AT Protocol search endpoint — no API key required.
Searches for quirky/unusual jewelry posts suitable for The Ugly section.
"""

import requests
from typing import List, Dict
from datetime import datetime


# Query variants to rotate across — each targets a different Ugly angle
BLUESKY_UGLY_QUERIES = [
    'unusual jewelry',
    'weird jewelry',
    'quirky jewelry',
    'bizarre jewelry',
    'funny jewelry',
    'novelty jewelry',
    'jewelry fail',
    'food jewelry',
]

BASE_URL = 'https://bsky.social/xrpc'


class BlueskyClient:
    """
    Read-only Bluesky search client using the public AT Protocol API.
    No credentials required for public post search.
    """

    USER_AGENT = 'JewelerNewsletter/1.0'

    def search_posts(
        self,
        query: str,
        limit: int = 10,
        sort: str = 'latest',
    ) -> List[Dict]:
        """Search public Bluesky posts for a query. Returns raw AT Protocol records."""
        try:
            response = requests.get(
                f'{BASE_URL}/app.bsky.feed.searchPosts',
                headers={'User-Agent': self.USER_AGENT},
                params={'q': query, 'limit': limit, 'sort': sort},
                timeout=10,
            )
            response.raise_for_status()
            return response.json().get('posts', [])
        except Exception as e:
            print(f'[Bluesky] Search error for "{query}": {e}')
            return []

    def search_for_ugly(
        self,
        max_results: int = 8,
        exclude_urls: List[str] = None,
    ) -> List[Dict]:
        """
        Search Bluesky for quirky/unusual jewelry content.
        Rotates across BLUESKY_UGLY_QUERIES and deduplicates by URL.
        Returns results normalized to the shared article schema.
        """
        exclude_urls = set(exclude_urls or [])
        seen_urls = set(exclude_urls)
        all_posts = []

        for query in BLUESKY_UGLY_QUERIES:
            if len(all_posts) >= max_results:
                break

            raw_posts = self.search_posts(query, limit=5, sort='latest')
            for post in raw_posts:
                normalized = self._normalize(post)
                if not normalized:
                    continue
                url = normalized['url']
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                all_posts.append(normalized)
                if len(all_posts) >= max_results:
                    break

            print(f'[Bluesky] "{query}" → {len(raw_posts)} posts')

        return all_posts[:max_results]

    def _normalize(self, post: dict) -> Dict | None:
        """Convert an AT Protocol post record to the shared article schema."""
        try:
            record = post.get('record', {})
            text = record.get('text', '').strip()
            if not text:
                return None

            # Build a stable URL from the author DID + record rkey
            author = post.get('author', {})
            handle = author.get('handle', '')
            uri = post.get('uri', '')  # e.g. at://did:.../app.bsky.feed.post/rkey
            rkey = uri.split('/')[-1] if uri else ''
            url = f'https://bsky.app/profile/{handle}/post/{rkey}' if handle and rkey else ''
            if not url:
                return None

            # Parse indexed timestamp
            indexed_at = post.get('indexedAt', '')
            try:
                published_at = datetime.fromisoformat(indexed_at.replace('Z', '+00:00')).strftime('%Y-%m-%d')
            except Exception:
                published_at = ''

            # Engagement signals
            like_count = post.get('likeCount', 0)
            repost_count = post.get('repostCount', 0)
            reply_count = post.get('replyCount', 0)

            return {
                'title': text[:120],           # first 120 chars as headline stand-in
                'url': url,
                'permalink': url,
                'publisher': f'@{handle} on Bluesky' if handle else 'Bluesky',
                'published_at': published_at,
                'snippet': text[:400],
                'thumbnail': '',
                'upvotes': like_count + repost_count,
                'num_comments': reply_count,
                'source_card': 'bluesky',
                'category': 'social',
                'impact': 'MEDIUM',
            }
        except Exception as e:
            print(f'[Bluesky] Normalize error: {e}')
            return None
