from datetime import datetime, timedelta
from typing import List, Dict
import aiohttp
import asyncio
from .base import BaseNewsSource

class NewsData(BaseNewsSource):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://newsdata.io/api/1/news"
        self.daily_limit = 190  # Buffer from 200
        
        # Override search groups for NewsData.io syntax
        self.search_groups = [
            # Major Companies
            "(Cheniere Energy) OR (EQT Corporation) OR (Kinder Morgan) OR (Williams Companies) OR (Dominion Energy)",
            
            # Production Regions
            "(Permian Basin AND gas) OR (Marcellus shale AND gas) OR (Haynesville AND gas) OR (Gulf Coast AND LNG)",
            
            # Market & Prices
            "(Henry Hub AND gas AND prices) OR (natural gas AND prices AND US)",
            
            # Infrastructure & Exports
            "(gas terminals AND US) OR (natural gas AND export AND US) OR (LNG AND exports AND United States)",
            
            # Regulatory
            "(FERC AND gas) OR (EIA AND gas AND report)",
            
            # Storage & Demand
            "(gas AND storage AND US) OR (gas AND demand AND winter) OR (gas AND demand AND summer)",
            
            # General Industry
            "(US AND natural gas) OR (LNG AND exports) OR (gas AND pipeline) OR (gas AND production AND US)"
        ]

    def _can_make_request(self) -> bool:
        now = datetime.now()
        if now - self.last_request_time > self.request_window:
            self.requests_made = 0
            return True
        return self.requests_made < self.daily_limit

    async def _fetch_for_query(self, session: aiohttp.ClientSession, query: str, page: int = 0) -> List[Dict]:
        """Fetch news for a specific query and page"""
        if not self._can_make_request():
            print("NewsData: Daily limit reached. Waiting for reset...")
            return []

        params = {
            "apikey": self.api_key,
            "q": query,
            "language": "en",
            "country": "us",
            "page": page
        }

        try:
            async with session.get(self.base_url, params=params) as response:
                self.requests_made += 1
                self.last_request_time = datetime.now()

                if response.status == 200:
                    data = await response.json()
                    
                    # Check if we have a next page
                    next_page = data.get('nextPage', None)
                    articles = data.get('results', [])
                    normalized_articles = [self.normalize_article(article) for article in articles]
                    
                    # If there's a next page and we haven't hit our limit, fetch it
                    if next_page and self._can_make_request():
                        await asyncio.sleep(1)  # Rate limiting
                        next_articles = await self._fetch_for_query(session, query, next_page)
                        normalized_articles.extend(next_articles)
                    
                    return normalized_articles
                else:
                    print(f"NewsData API error: {response.status}")
                    return []

        except Exception as e:
            print(f"NewsData Error for '{query[:50]}...': {str(e)}")
            return []

    async def fetch_news(self) -> List[Dict]:
        """Fetch news from NewsData API"""
        async with aiohttp.ClientSession() as session:
            all_articles = []
            
            for search_group in self.search_groups:
                if not self._can_make_request():
                    print("NewsData: Daily limit reached. Stopping further requests.")
                    break
                    
                articles = await self._fetch_for_query(session, search_group)
                all_articles.extend(articles)
                await asyncio.sleep(1)  # Rate limiting

            return all_articles

    def normalize_article(self, article: Dict) -> Dict:
        """Normalize NewsData article format"""
        return {
            'title': article.get('title', 'No title available'),
            'content': article.get('description', 'No content available'),
            'url': article.get('link', ''),
            'source': article.get('source_id', 'Unknown source'),
            'published_date': article.get('pubDate', datetime.now(datetime.UTC).isoformat()),
            'image_url': article.get('image_url', None)
        }

    def get_remaining_requests(self) -> int:
        if not self._can_make_request():
            return 0
        return self.daily_limit - self.requests_made