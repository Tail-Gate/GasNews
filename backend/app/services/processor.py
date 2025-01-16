from typing import List, Dict

class NewsProcessor:
    def process_articles(self, articles: List[Dict]) -> List[Dict]:
        processed_articles = []
        seen_urls = set()
        
        for article in articles:
            if article['url'] not in seen_urls:
                # Handle possible null values with defaults
                processed_article = {
                    'title': article['title'] or 'No title available',
                    'content': article['description'] or 'No content available',  # Using description instead of content
                    'url': article['url'],
                    'source': article['source']['name'] if article['source'] else 'Unknown source',
                    'published_date': article['publishedAt'],
                    'image_url': article.get('urlToImage')
                }
                processed_articles.append(processed_article)
                seen_urls.add(article['url'])
                
        return processed_articles