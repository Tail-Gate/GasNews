from typing import List, Dict, Set
import asyncio
from datetime import datetime
from difflib import SequenceMatcher
import logging

class NewsAggregator:
    def __init__(self, news_sources: List):
        self.news_sources = news_sources
        self._seen_urls: Set[str] = set()
        self._seen_titles: Dict[str, datetime] = {}

    def _is_duplicate(self, article: Dict, title_threshold: float = 0.85) -> bool:
        """
        Check if an article is a duplicate based on URL or title similarity
        """
        url = article['url']
        title = article['title'].lower()
        pub_date = datetime.fromisoformat(article['published_date'].replace('Z', '+00:00'))

        # Check URL
        if url in self._seen_urls:
            return True

        # Check title similarity with recent articles (within 24 hours)
        for existing_title, existing_date in list(self._seen_titles.items()):
            # Remove old titles from memory
            if (pub_date - existing_date).days >= 1:
                del self._seen_titles[existing_title]
                continue

            # Check title similarity
            similarity = SequenceMatcher(None, title, existing_title.lower()).ratio()
            if similarity >= title_threshold:
                return True

        # Not a duplicate - add to seen items
        self._seen_urls.add(url)
        self._seen_titles[title] = pub_date
        return False

    async def _fetch_from_source(self, source) -> List[Dict]:
        """Fetch news from a single source with error handling"""
        try:
            articles = await source.fetch_news()
            logging.info(f"Fetched {len(articles)} articles from {source.__class__.__name__}")
            return articles
        except Exception as e:
            logging.error(f"Error fetching from {source.__class__.__name__}: {str(e)}")
            return []

    async def fetch_all_news(self) -> List[Dict]:
        """
        Fetch and aggregate news from all sources
        """
        # Reset seen articles
        self._seen_urls.clear()
        self._seen_titles.clear()

        # Fetch from all sources concurrently
        tasks = [self._fetch_from_source(source) for source in self.news_sources]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine and deduplicate articles
        unique_articles = []
        for source_articles in all_results:
            if isinstance(source_articles, Exception):
                continue
                
            for article in source_articles:
                if not self._is_duplicate(article):
                    unique_articles.append(article)

        # Sort by published date (newest first)
        unique_articles.sort(
            key=lambda x: datetime.fromisoformat(x['published_date'].replace('Z', '+00:00')),
            reverse=True
        )

        return unique_articles

    def get_source_status(self) -> List[Dict]:
        """
        Get status of all news sources
        """
        return [
            {
                'name': source.__class__.__name__,
                'remaining_requests': source.get_remaining_requests()
            }
            for source in self.news_sources
        ]