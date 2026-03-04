"""
Reddit API Client

Uses Reddit's OAuth client_credentials flow (read-only, no user login needed).
Searches jewelry-relevant subreddits with proper time filtering.
"""

import os
import json
import requests
import base64
from typing import List, Dict
from datetime import datetime


# Subreddits to search for The Ugly quirky/unusual jewelry content
# Paired with the query best suited for each community's context
UGLY_SUBREDDIT_QUERIES = [
    ('jewelry',               'unusual weird quirky bizarre jewelry'),
    ('Justrolledintotheshop', 'jewelry unusual bizarre weird repair'),
    ('mildlyinteresting',     'jewelry unusual quirky'),
    ('DIYjewelry',            'unusual quirky wearable art bizarre'),
    ('ATBGE',                 'jewelry ring necklace bracelet'),       # "Awful Taste But Great Execution" — high signal
    ('ThriftStoreHauls',      'jewelry unusual vintage weird find'),
    ('whatisthisthing',       'jewelry ring necklace bracelet'),       # Mystery jewelry ID posts
]

# Keep legacy constants for backward compatibility
UGLY_SUBREDDITS = [sr for sr, _ in UGLY_SUBREDDIT_QUERIES]
UGLY_SEARCH_TERMS = [
    'unusual jewelry',
    'weird jewelry',
    'quirky jewelry',
    'novelty jewelry',
    'food jewelry',
    'funny jewelry',
    'bizarre jewelry',
    'wearable art',
]


class RedditClient:
    """Client for Reddit API — read-only, uses client_credentials OAuth flow."""

    BASE_URL = 'https://oauth.reddit.com'
    TOKEN_URL = 'https://www.reddit.com/api/v1/access_token'
    USER_AGENT = 'JewelerNewsletter/1.0 (by JewelryProtector)'

    def __init__(self, client_id: str = None, client_secret: str = None):
        self.client_id = client_id or os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('REDDIT_CLIENT_SECRET')
        self._access_token = None

        if self.client_id and self.client_secret:
            print("[OK] Reddit initialized")
        else:
            print("[WARNING] REDDIT_CLIENT_ID or REDDIT_CLIENT_SECRET not found - Reddit search disabled")

    def is_available(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _get_access_token(self) -> str:
        """Fetch a client_credentials access token. Cached per instance."""
        if self._access_token:
            return self._access_token

        credentials = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()

        response = requests.post(
            self.TOKEN_URL,
            headers={
                'Authorization': f'Basic {credentials}',
                'User-Agent': self.USER_AGENT,
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            data={'grant_type': 'client_credentials'},
            timeout=10
        )
        response.raise_for_status()
        self._access_token = response.json()['access_token']
        return self._access_token

    def _reddit_time_filter(self, time_window: str) -> str:
        """Map our time_window codes to Reddit's t= filter values."""
        return {
            '7d': 'week',
            '15d': 'month',
            '30d': 'month',
            '90d': 'year',
        }.get(time_window, 'month')

    def search_subreddit(
        self,
        subreddit: str,
        query: str,
        time_filter: str = 'month',
        max_results: int = 5
    ) -> List[Dict]:
        """Search within a specific subreddit."""
        token = self._get_access_token()
        response = requests.get(
            f"{self.BASE_URL}/r/{subreddit}/search",
            headers={
                'Authorization': f'Bearer {token}',
                'User-Agent': self.USER_AGENT,
            },
            params={
                'q': query,
                't': time_filter,
                'sort': 'top',
                'restrict_sr': '1',  # limit to this subreddit
                'limit': max_results,
                'type': 'link',
            },
            timeout=10
        )
        response.raise_for_status()
        return self._parse_posts(response.json(), subreddit)

    def search_for_ugly(
        self,
        time_window: str = '30d',
        max_results: int = 8,
        exclude_urls: List[str] = None
    ) -> List[Dict]:
        """
        Search for quirky/unusual jewelry content across relevant subreddits.
        Returns results normalized to the shared article schema.
        """
        if not self.is_available():
            print("[Reddit] Not configured, skipping")
            return []

        exclude_urls = set(exclude_urls or [])
        time_filter = self._reddit_time_filter(time_window)
        all_posts = []
        seen_urls = set(exclude_urls)

        try:
            token = self._get_access_token()
        except Exception as e:
            print(f"[Reddit] Auth failed: {e}")
            return []

        for subreddit, query in UGLY_SUBREDDIT_QUERIES:
            if len(all_posts) >= max_results:
                break
            try:
                posts = self.search_subreddit(subreddit, query, time_filter, max_results=4)
                for post in posts:
                    url = post.get('url', '')
                    if url and url not in seen_urls:
                        all_posts.append(post)
                        seen_urls.add(url)
                print(f"[Reddit] r/{subreddit} '{query[:40]}' → {len(posts)} posts")
            except Exception as e:
                print(f"[Reddit] Error searching r/{subreddit}: {e}")
                continue

        return all_posts[:max_results]

    def _parse_posts(self, data: dict, subreddit: str) -> List[Dict]:
        """Parse Reddit API response into shared article schema."""
        posts = []
        children = data.get('data', {}).get('children', [])

        for child in children:
            post = child.get('data', {})
            if not post:
                continue

            # Skip posts with no useful content
            title = post.get('title', '').strip()
            if not title:
                continue

            # Build the URL — prefer external link, fall back to Reddit thread
            url = post.get('url', '')
            permalink = 'https://www.reddit.com' + post.get('permalink', '')

            # If the post links to Reddit itself (self post or gallery), use the permalink
            if not url or 'reddit.com' in url or url.startswith('/'):
                url = permalink

            # Build a snippet from the post text or just the title
            selftext = post.get('selftext', '').strip()
            snippet = selftext[:400] if selftext and selftext != '[removed]' else title

            # Convert Unix timestamp to date string
            created_utc = post.get('created_utc', 0)
            try:
                published_at = datetime.utcfromtimestamp(created_utc).strftime('%Y-%m-%d')
            except Exception:
                published_at = ''

            # Build thumbnail URL if available (useful for visual content)
            thumbnail = post.get('thumbnail', '')
            if thumbnail in ('self', 'default', 'nsfw', 'spoiler', ''):
                thumbnail = ''

            posts.append({
                'title': title,
                'url': url,
                'permalink': permalink,
                'publisher': f'r/{post.get("subreddit", subreddit)}',
                'published_at': published_at,
                'snippet': snippet,
                'thumbnail': thumbnail,
                'upvotes': post.get('score', 0),
                'num_comments': post.get('num_comments', 0),
                'source_card': 'reddit',
                'category': 'social',
                'impact': 'MEDIUM',
            })

        return posts
