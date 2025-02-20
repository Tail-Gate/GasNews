from datetime import datetime, timezone, timedelta
from typing import List, Dict
import aiohttp
import asyncio
from .base import BaseNewsSource

class CurrentsSource(BaseNewsSource):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.currentsapi.services/v1/search"
        self.daily_limit = 550  # Buffer from 600
        
        # Override search groups for Currents API syntax
        self.search_groups = [
            # Major Companies
            "Cheniere Energy,EQT Corporation,Kinder Morgan,Williams Companies,Dominion Energy",
            
            # Production Regions
            "Permian Basin,Marcellus shale,Haynesville,Gulf Coast LNG",
            
            # Market & Prices
            "Henry Hub gas prices,natural gas prices,US market",
            
            # Infrastructure & Exports
            "natural gas terminals,LNG exports,United States",
            
            # Regulatory
            "FERC gas,EIA gas report",
            
            # Storage & Demand
            "natural gas storage,gas demand",
            
            # General Industry
            "US natural gas,LNG exports,gas pipeline,gas production"
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
            print("Currents API: Daily limit reached. Waiting for reset...")
            return []

        # Calculate date range (last 7 days)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)

        params = {
            "apiKey": self.api_key,
            "keywords": query,
            "language": "en",
            "country": "US",  # Focus on US news
            "start_date": start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "end_date": end_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "type": "1",  # News type
            "page_size": 200  # Maximum results per request
        }

        try:
            async with session.get(self.base_url, params=params) as response:
                self.requests_made += 1
                self.last_request_time = datetime.now()

                if response.status == 200:
                    data = await response.json()
                    if 'news' in data:
                        return [self.normalize_article(article) for article in data['news']]
                    else:
                        print(f"Currents API: No news field in response")
                        return []
                else:
                    error_data = await response.text()
                    print(f"Currents API error {response.status}: {error_data}")
                    return []

        except Exception as e:
            print(f"Currents API Error for '{query[:50]}...': {str(e)}")
            return []

    async def fetch_news(self) -> List[Dict]:
        """Fetch news from Currents API"""
        async with aiohttp.ClientSession() as session:
            all_articles = []
            
            for search_group in self.search_groups:
                if not self._can_make_request():
                    print("Currents API: Daily limit reached. Stopping further requests.")
                    break
                    
                articles = await self._fetch_for_query(session, search_group)
                all_articles.extend(articles)
                await asyncio.sleep(1)  # Rate limiting

            return all_articles

    def normalize_article(self, article: Dict) -> Dict:
        """Normalize Currents API article format"""
        return {
            'title': article.get('title', 'No title available'),
            'content': article.get('description', 'No content available'),
            'url': article.get('url', ''),
            'source': article.get('author', 'Unknown source'),
            'published_date': article.get('published', datetime.now(timezone.utc).isoformat()),
            'image_url': article.get('image', None)
        }

    def get_remaining_requests(self) -> int:
        if not self._can_make_request():
            return 0
        return self.daily_limit - self.requests_made