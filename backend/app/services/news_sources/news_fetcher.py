from datetime import datetime, timedelta
from typing import List, Dict
from newsapi import NewsApiClient
import asyncio
from functools import lru_cache

class NewsFetcher:
    def __init__(self, api_key: str):
        self.news_api = NewsApiClient(api_key=api_key)
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
        self.last_request_time = datetime.now()
        self.requests_made = 0
        self.daily_limit = 95  # Setting slightly below 100 for safety
        self.request_window = timedelta(hours=24)

    def _can_make_request(self) -> bool:
        """Check if we can make another API request within our limits"""
        now = datetime.now()
        if now - self.last_request_time > self.request_window:
            self.requests_made = 0
            return True
        return self.requests_made < self.daily_limit

    @lru_cache(maxsize=16)
    def _get_cached_news(self, search_term: str, date_str: str) -> Dict:
        """Cache API responses to avoid redundant calls"""
        return self.news_api.get_everything(
            q=search_term,
            language='en',
            sort_by='publishedAt',
            from_param=date_str
        )

    async def fetch_news(self) -> List[Dict]:
        articles = []
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        for search_group in self.search_groups:
            if not self._can_make_request():
                print("Daily API limit reached. Waiting for reset...")
                # Calculate time until reset
                next_window = self.last_request_time + self.request_window
                wait_time = (next_window - datetime.now()).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                self.requests_made = 0
            
            try:
                # Make API request with caching
                news_response = self._get_cached_news(search_group, from_date)
                
                if news_response['status'] == 'ok':
                    self.requests_made += 1
                    self.last_request_time = datetime.now()
                    
                    for article in news_response['articles']:
                        if article.get('content') is None:
                            article['content'] = "Content not available"
                        articles.append(article)
                    
                    # Add delay between requests
                    await asyncio.sleep(1)
                
            except Exception as e:
                error_msg = str(e)
                print(f"Error fetching news for group '{search_group[:50]}...': {error_msg}")
                
                if 'rateLimited' in error_msg:
                    print("Rate limit hit. Pausing requests...")
                    await asyncio.sleep(300)  # 5 minute cooldown
                    continue
                    
        return articles

    def get_remaining_requests(self) -> int:
        """Return number of remaining API requests in current window"""
        if not self._can_make_request():
            return 0
        return self.daily_limit - self.requests_made