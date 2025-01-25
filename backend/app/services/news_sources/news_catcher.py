from datetime import datetime, timedelta
from typing import List, Dict
import aiohttp
import asyncio
from .base import BaseNewsSource

class NewsDataSource(BaseNewsSource):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.newscatcherapi.com/v2/search"
        self.daily_limit = 950  # Keep buffer from 1000
        self.headers = {
            "x-api-key": api_key
        }

    def _can_make_request(self) -> bool:
        now = datetime.now()
        if now - self.last_request_time > self.request_window:
            self.requests_made = 0
            return True
        return self.requests_made < self.daily_limit

    async def _fetch_for_query(self, session: aiohttp.ClientSession, query: str) -> List[Dict]:
        """Fetch news for a specific query"""
        if not self._can_make_request():
            print("NewsCatcher: Daily limit reached. Waiting for reset...")
            return []

        params = {
            "q": query,
            "lang": "en",
            "sort_by": "relevancy",
            "page_size": 100
        }

        try:
            async with session.get(self.base_url, headers=self.headers, params=params) as response:
                self.requests_made += 1
                self.last_request_time = datetime.now()

                if response.status == 200:
                    data = await response.json()
                    articles = data.get('articles', [])
                    return [self.normalize_article(article) for article in articles]
                else:
                    print(f"NewsCatcher API error: {response.status}")
                    return []

        except Exception as e:
            print(f"NewsCatcher Error for '{query[:50]}...': {str(e)}")
            return []

    async def fetch_news(self) -> List[Dict]:
        """Fetch news from NewsCatcher API"""
        async with aiohttp.ClientSession() as session:
            all_articles = []
            
            for search_group in self.search_groups:
                articles = await self._fetch_for_query(session, search_group)
                all_articles.extend(articles)
                await asyncio.sleep(1)  # Rate limiting

            return all_articles

    def normalize_article(self, article: Dict) -> Dict:
        """Normalize NewsCatcher article format"""
        return {
            'title': article.get('title', 'No title available'),
            'content': article.get('summary', 'No content available'),
            'url': article.get('link', ''),
            'source': article.get('rights', 'Unknown source'),
            'published_date': article.get('published_date', datetime.utcnow().isoformat()),
            'image_url': article.get('media', None)
        }

    def get_remaining_requests(self) -> int:
        if not self._can_make_request():
            return 0
        return self.daily_limit - self.requests_made