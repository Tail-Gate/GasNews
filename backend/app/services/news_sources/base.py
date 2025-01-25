from abc import ABC, abstractmethod
from typing import List, Dict
from datetime import datetime, timedelta
import asyncio

class BaseNewsSource(ABC):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.last_request_time = datetime.now()
        self.requests_made = 0
        self.request_window = timedelta(hours=24)
        
        # Common search terms across all sources
        self.search_groups = [
            # Major Companies & Infrastructure
            "Cheniere Energy OR EQT Corporation OR Kinder Morgan OR Williams Companies OR Dominion Energy",
            # Key Production Regions
            "Permian Basin gas OR Marcellus shale gas OR Haynesville gas OR Gulf Coast LNG",
            # Market & Prices
            "Henry Hub gas prices OR Northeast gas prices OR Texas gas market OR natural gas prices US",
            # Infrastructure & Exports
            "US gas terminals OR US natural gas export OR Jones Act LNG OR LNG exports United States",
            # Regulatory & Reports
            "FERC gas OR US gas regulations OR EIA gas report",
            # Storage & Demand
            "US gas storage OR US winter gas demand OR US summer cooling gas OR US gas storage report",
            # General Industry
            "US natural gas OR American LNG exports OR US gas pipeline OR domestic gas production OR US energy security"
        ]

    @abstractmethod
    async def fetch_news(self) -> List[Dict]:
        """Fetch news articles from the source"""
        pass

    @abstractmethod
    def _can_make_request(self) -> bool:
        """Check if we can make another API request within limits"""
        pass

    @abstractmethod
    def get_remaining_requests(self) -> int:
        """Get number of remaining API requests in current window"""
        pass

    def normalize_article(self, article: Dict) -> Dict:
        """Normalize article format across different sources"""
        return {
            'title': article.get('title', 'No title available'),
            'content': article.get('content') or article.get('description', 'No content available'),
            'url': article.get('url', ''),
            'source': article.get('source', {}).get('name', 'Unknown source'),
            'published_date': article.get('publishedAt') or article.get('published_date', datetime.now(datetime.UTC).isoformat()),
            'image_url': article.get('urlToImage') or article.get('image_url'),
        }