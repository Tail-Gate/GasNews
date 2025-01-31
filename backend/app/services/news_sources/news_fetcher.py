from datetime import datetime, timedelta
import asyncio
from typing import List, Dict
from newsapi import NewsApiClient
from functools import lru_cache
from .base import BaseNewsSource


class NewsAPISource(BaseNewsSource):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.news_api = NewsApiClient(api_key=api_key)
        self.daily_limit = 95  # Setting slightly below 100 for safety

    def _can_make_request(self) -> bool:
        now = datetime.now()
        if now - self.last_request_time > self.request_window:
            self.requests_made = 0
            return True
        return self.requests_made < self.daily_limit

    @lru_cache(maxsize=16)
    def _get_cached_news(self, search_term: str, date_str: str) -> Dict:
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
                print("NewsAPI: Daily limit reached. Waiting for reset...")
                next_window = self.last_request_time + self.request_window
                wait_time = (next_window - datetime.now()).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                self.requests_made = 0
            
            try:
                news_response = self._get_cached_news(search_group, from_date)
                
                if news_response['status'] == 'ok':
                    self.requests_made += 1
                    self.last_request_time = datetime.now()
                    
                    for article in news_response['articles']:
                        normalized_article = self.normalize_article(article)
                        articles.append(normalized_article)
                    
                    await asyncio.sleep(1)  # Rate limiting
                
            except Exception as e:
                error_msg = str(e)
                print(f"NewsAPI Error for '{search_group[:50]}...': {error_msg}")
                
                if 'rateLimited' in error_msg:
                    print("Rate limit hit. Pausing requests...")
                    await asyncio.sleep(300)  # 5 minute cooldown
                    continue
                    
        return articles

    def get_remaining_requests(self) -> int:
        if not self._can_make_request():
            return 0
        return self.daily_limit - self.requests_made