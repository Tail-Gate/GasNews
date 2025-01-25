from datetime import datetime, timedelta
from typing import List, Dict
import aiohttp
import asyncio
from .base import BaseNewsSource

class GNewsSource(BaseNewsSource):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://gnews.io/api/v4/search"
        self.daily_limit = 95  # Buffer from 100
        
        
        # Override search groups for GNews syntax
        self.search_groups = [
            # Major Companies
            '"Cheniere Energy" OR "EQT Corporation" OR "Kinder Morgan" OR "Williams Companies" OR "Dominion Energy"',
            
            # Production Regions (with context)
            '"Permian Basin" natural gas OR "Marcellus shale" gas OR "Haynesville" gas OR "Gulf Coast" LNG',
            
            # Market & Prices (simplified for GNews)
            '"Henry Hub" gas price OR "natural gas prices" US market',
            
            # Infrastructure & Exports
            '"natural gas terminals" US OR "LNG exports" United States',
            
            # Regulatory
            '"FERC" natural gas OR "EIA gas report"',
            
            # Storage & Demand
            '"natural gas storage" US OR "gas demand" winter OR "gas demand" summer',
            
            # General Industry
            '"US natural gas" OR "LNG exports" OR "gas pipeline" OR "gas production" US'
        ]

    def _can_make_request(self) -> bool:
        now = datetime.now()
        if now - self.last_request_time > self.request_window:
            self.requests_made = 0
            return True
        return self.requests_made < self.daily_limit

    async def _fetch_for_query(self, session: aiohttp.ClientSession, query: str) -> List[Dict]:
        """Fetch news for a specific query"""
        if not self._can_make_request():
            print("GNews: Daily limit reached. Waiting for reset...")
            return []

        params = {
            "q": query,
            "lang": "en",
            "country": "us",  # Focus on US news
            "max": 50,  # Max articles per request
            "apikey": self.api_key
        }

        try:
            async with session.get(self.base_url, params=params) as response:
                self.requests_made += 1
                self.last_request_time = datetime.now()

                if response.status == 200:
                    data = await response.json()
                    return [self.normalize_article(article) for article in data.get('articles', [])]
                else:
                    print(f"GNews API error: {response.status}")
                    return []

        except Exception as e:
            print(f"GNews Error for '{query[:50]}...': {str(e)}")
            await asyncio.sleep(1)  # Brief pause on error
            return []

    async def fetch_news(self) -> List[Dict]:
        """Fetch news from GNews API"""
        async with aiohttp.ClientSession() as session:
            all_articles = []
            
            for search_group in self.search_groups:
                if not self._can_make_request():
                    print("GNews: Daily limit reached. Stopping further requests.")
                    break
                    
                articles = await self._fetch_for_query(session, search_group)
                all_articles.extend(articles)
                await asyncio.sleep(2)  # Longer delay between requests for GNews

            return all_articles

    def normalize_article(self, article: Dict) -> Dict:
        """Normalize GNews article format"""
        return {
            'title': article.get('title', 'No title available'),
            'content': article.get('description', 'No content available'),
            'url': article.get('url', ''),
            'source': article.get('source', {}).get('name', 'Unknown source'),
            'published_date': article.get('publishedAt', datetime.utcnow().isoformat()),
            'image_url': article.get('image', None)
        }

    def get_remaining_requests(self) -> int:
        if not self._can_make_request():
            return 0
        return self.daily_limit - self.requests_made