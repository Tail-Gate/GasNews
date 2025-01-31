import asyncio
import os
from dotenv import load_dotenv
from datetime import datetime
import json

from services.news_sources import (
    NewsAPISource, 
    NewsDataSource,
    CurrentsSource,
    GNewsSource,
    NewsData,
    NewsAggregator
)

async def test_source(source, name):
    print(f"\n{'='*50}")
    print(f"Testing {name}...")
    print(f"{'='*50}")
    
    try:
        # Print remaining requests before fetching
        remaining = source.get_remaining_requests()
        print(f"Remaining requests: {remaining}")
        
        start = datetime.now()
        articles = await source.fetch_news()
        duration = datetime.now() - start
        
        print(f"\nResults:")
        print(f"✓ Retrieved {len(articles)} articles in {duration.seconds}s")
        
        if articles:
            print("\nFirst article details:")
            article = articles[0]
            for key, value in article.items():
                print(f"{key}: {value[:100]}..." if isinstance(value, str) else f"{key}: {value}")
        
        # Print remaining requests after fetching
        remaining = source.get_remaining_requests()
        print(f"\nRemaining requests after fetch: {remaining}")
        
        return articles
    except Exception as e:
        print(f"✕ Error testing {name}: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return []

async def main():
    load_dotenv()
    
    # First print all API keys (masked) to verify they're loaded
    api_keys = {
        "NEWS_API_KEY": os.getenv("NEWS_API_KEY"),
        "NEWSCATCHER_API_KEY": os.getenv("NEWSCATCHER_API_KEY"),
        "CURRENTS_API_KEY": os.getenv("CURRENTS_API_KEY"),
        "GNEWS_API_KEY": os.getenv("GNEWS_API_KEY"),
        "NEWSDATA_API_KEY": os.getenv("NEWSDATA_API_KEY")
    }
    
    print("API Keys loaded:")
    for key, value in api_keys.items():
        if value:
            masked = value[:4] + '*' * (len(value)-8) + value[-4:]
            print(f"{key}: {masked}")
        else:
            print(f"{key}: Not found!")
    
    sources = [
        (NewsAPISource(api_keys["NEWS_API_KEY"]), "NewsAPI"),
        (NewsDataSource(api_keys["NEWSCATCHER_API_KEY"]), "NewsCatcher"),
        (CurrentsSource(api_keys["CURRENTS_API_KEY"]), "Currents"),
        (GNewsSource(api_keys["GNEWS_API_KEY"]), "GNews"),
        (NewsData(api_keys["NEWSDATA_API_KEY"]), "NewsData")
    ]

    # Test each source with delay between tests
    for source, name in sources:
        await test_source(source, name)
        print("\nWaiting 5 seconds before next test...")
        await asyncio.sleep(5)

    print("\nTesting Aggregator...")
    aggregator = NewsAggregator([s[0] for s in sources])
    
    start = datetime.now()
    all_articles = await aggregator.fetch_all_news()
    duration = datetime.now() - start
    
    print(f"\nAggregator Results:")
    print(f"✓ Total articles: {len(all_articles)}")
    print(f"✓ Total time: {duration.seconds}s")
    
    # Check for duplicates
    urls = [a['url'] for a in all_articles]
    unique_urls = set(urls)
    print(f"✓ Unique articles: {len(unique_urls)}")
    
    if len(urls) != len(unique_urls):
        print(f"⚠ Found {len(urls) - len(unique_urls)} duplicate URLs")
    else:
        print("✓ No duplicate URLs found")

if __name__ == "__main__":
    asyncio.run(main())