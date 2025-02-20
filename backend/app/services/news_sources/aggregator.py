from typing import List, Dict, Set, Optional, Any
import asyncio
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher
import logging
from collections import defaultdict
import hashlib

class NewsAggregator:
    def __init__(self, news_sources: List[Any]):
        """
        Initialize the NewsAggregator with multiple news sources
        
        Args:
            news_sources: List of news source instances
        """
        self.news_sources = news_sources
        self._seen_urls: Set[str] = set()
        self._seen_titles: Dict[str, datetime] = {}
        self._content_hashes: Set[str] = set()
        self._source_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"success": 0, "errors": 0})
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _compute_content_hash(self, content: str) -> str:
        """Generate a hash of the article content for duplicate detection"""
        return hashlib.md5(content.lower().encode()).hexdigest()

    def _is_duplicate(self, article: Dict, title_threshold: float = 0.85, content_threshold: float = 0.95) -> bool:
        """
        Check if an article is a duplicate based on URL, title similarity, or content
        
        Args:
            article: Article dictionary containing title, url, and content
            title_threshold: Similarity threshold for title comparison
            content_threshold: Similarity threshold for content comparison
        
        Returns:
            bool: True if article is a duplicate, False otherwise
        """
        try:
            url = article['url'].strip()
            title = article['title'].lower().strip()
            content = article.get('content', '').strip()
            pub_date = datetime.fromisoformat(article['published_date'].replace('Z', '+00:00'))
            
            # Check URL
            if url in self._seen_urls:
                return True
                
            # Generate content hash if content exists
            if content:
                content_hash = self._compute_content_hash(content)
                if content_hash in self._content_hashes:
                    return True
                
            # Check title similarity with recent articles (within 24 hours)
            current_time = datetime.now(pub_date.tzinfo)
            for existing_title, existing_date in list(self._seen_titles.items()):
                # Remove old titles from memory
                if (current_time - existing_date).days >= 1:
                    del self._seen_titles[existing_title]
                    continue
                    
                # Check title similarity
                similarity = SequenceMatcher(None, title, existing_title.lower()).ratio()
                if similarity >= title_threshold:
                    return True
                    
            # Not a duplicate - add to seen items
            self._seen_urls.add(url)
            self._seen_titles[title] = pub_date
            if content:
                self._content_hashes.add(self._compute_content_hash(content))
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error in duplicate detection: {str(e)}")
            return False  # Err on the side of inclusion

    async def _fetch_from_source(self, source: Any) -> List[Dict]:
        """
        Fetch news from a single source with error handling and statistics
        
        Args:
            source: News source instance
            
        Returns:
            List of article dictionaries
        """
        source_name = source.__class__.__name__
        try:
            articles = await source.fetch_news()
            valid_articles = []
            
            for article in articles:
                try:
                    # Validate required fields
                    required_fields = ['title', 'url', 'published_date']
                    if not all(field in article and article[field] for field in required_fields):
                        continue
                        
                    # Add source tracking
                    article['source_name'] = source_name
                    article['fetched_at'] = datetime.now(timezone.utc).isoformat()
                    
                    valid_articles.append(article)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing article from {source_name}: {str(e)}")
                    continue
                    
            self._source_stats[source_name]["success"] += 1
            self.logger.info(f"Fetched {len(valid_articles)} articles from {source_name}")
            return valid_articles
            
        except Exception as e:
            self._source_stats[source_name]["errors"] += 1
            self.logger.error(f"Error fetching from {source_name}: {str(e)}")
            return []

    async def fetch_all_news(self, timeout: int = 30) -> List[Dict]:
        """
        Fetch and aggregate news from all sources with timeout
        
        Args:
            timeout: Maximum seconds to wait for all sources
            
        Returns:
            List of unique, sorted articles
        """
        # Reset seen articles
        self._seen_urls.clear()
        self._seen_titles.clear()
        self._content_hashes.clear()
        
        # Fetch from all sources concurrently with timeout
        try:
            tasks = [self._fetch_from_source(source) for source in self.news_sources]
            all_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout after {timeout} seconds waiting for news sources")
            all_results = []
        
        # Combine and deduplicate articles
        unique_articles = []
        for source_articles in all_results:
            if isinstance(source_articles, Exception):
                self.logger.error(f"Source error: {str(source_articles)}")
                continue
                
            for article in source_articles:
                if not self._is_duplicate(article):
                    unique_articles.append(article)
        
        # Sort by published date (newest first)
        unique_articles.sort(
            key=lambda x: datetime.fromisoformat(
                x['published_date'].replace('Z', '+00:00')
            ),
            reverse=True
        )
        
        self.logger.info(f"Aggregated {len(unique_articles)} unique articles from {len(self.news_sources)} sources")
        return unique_articles

    def get_source_status(self) -> List[Dict]:
        """
        Get status of all news sources including success/error stats
        
        Returns:
            List of source status dictionaries
        """
        return [
            {
                'name': source.__class__.__name__,
                'remaining_requests': source.get_remaining_requests(),
                'success_count': self._source_stats[source.__class__.__name__]["success"],
                'error_count': self._source_stats[source.__class__.__name__]["errors"]
            }
            for source in self.news_sources
        ]

    def clear_cache(self, older_than_hours: int = 24) -> None:
        """
        Clear cached data older than specified hours
        
        Args:
            older_than_hours: Number of hours after which to clear cached data
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)
        
        # Clear old titles
        self._seen_titles = {
            title: date for title, date in self._seen_titles.items()
            if date > cutoff_time
        }
        
        # Reset other caches if they're older than the cutoff
        if not self._seen_titles:  # If all titles were old
            self._seen_urls.clear()
            self._content_hashes.clear()
            self.logger.info("Cleared all cached data")