from typing import List, Dict

class NewsProcessor:
    def process_articles(self, articles: List[Dict]) -> List[Dict]:
        processed_articles = []
        seen_urls = set()
        
        for article in articles:
            if not article.get('url'):
                continue  # Skip articles without URL
                
            if article['url'] not in seen_urls:
                # Handle different API response formats
                content = article.get('content') or article.get('description') or 'No content available'
                source_name = article.get('source', {}).get('name') if isinstance(article.get('source'), dict) else article.get('source')
                
                processed_article = {
                    'title': article.get('title') or 'No title available',
                    'content': content,
                    'url': article['url'],
                    'source': source_name or 'Unknown source',
                    'published_date': article.get('publishedAt') or article.get('published_date', datetime.now().isoformat()),
                    'image_url': article.get('urlToImage') or article.get('image_url')
                }
                processed_articles.append(processed_article)
                seen_urls.add(article['url'])
            
        return processed_articles