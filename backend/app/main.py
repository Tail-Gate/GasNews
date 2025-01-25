from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from .database import engine, SessionLocal
from . import models, crud, schemas
from typing import List
from .services.news_sources import NewsAPISource,NewsDataSource,CurrentsSource,GNewsSource,NewsData,NewsAggregator
from .services.processor import NewsProcessor
from .services.notifier import NewsNotifier
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize news sources
news_sources = [
    NewsAPISource(os.getenv("NEWS_API_KEY")),
    NewsDataSource(os.getenv("NEWSCATCHER_API_KEY")),
    CurrentsSource(os.getenv("CURRENTS_API_KEY")),
    GNewsSource(os.getenv("GNEWS_API_KEY")),
    NewsData(os.getenv("NEWSDATA_API_KEY"))
]

# Initialize services
news_aggregator = NewsAggregator(news_sources)
news_processor = NewsProcessor()
news_notifier = NewsNotifier()

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Your existing root endpoints
@app.get("/")
async def root():
    return {"message": "Natural Gas Source news"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Article endpoints
@app.get("/articles/")
def read_articles(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    articles = crud.get_articles(db, skip=skip, limit=limit)
    return articles

@app.get("/articles/{article_id}")
def read_article(article_id: int, db: Session = Depends(get_db)):
    article = crud.get_article(db, article_id=article_id)
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found")
    return article

# User endpoints
@app.post("/users/", response_model=schemas.UserResponse)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.create_user(db, username=user.username, hashed_password=user.password, email=user.email)
    return db_user

@app.get("/users/{user_id}", response_model=schemas.UserResponse)
def read_user(user_id: int, db: Session = Depends(get_db)):
    user = crud.get_user(db, user_id=user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# Bookmark endpoints
@app.post("/bookmarks/{article_id}", response_model=schemas.BookmarkResponse)
def create_bookmark(
    article_id: int, 
    bookmark_data: schemas.BookmarkRequest,
    db: Session = Depends(get_db)
):
    return crud.create_bookmark(db, user_id=bookmark_data.user_id, article_id=article_id)

@app.get("/bookmarks/{user_id}", response_model=List[schemas.BookmarkResponse])
def read_user_bookmarks(user_id: int, db: Session = Depends(get_db)):
    return crud.get_user_bookmarks(db, user_id=user_id)

@app.delete("/bookmarks/{article_id}")
def delete_user_bookmark(article_id: int, user_id: int, db: Session = Depends(get_db)):
    success = crud.delete_bookmark(db, user_id=user_id, article_id=article_id)
    if not success:
        raise HTTPException(status_code=404, detail="Bookmark not found")
    return {"status": "success"}

# News endpoints
@app.get("/fetch-news/", response_model=List[schemas.ArticleResponse])
async def fetch_news(db: Session = Depends(get_db)):
    try:
        # 1. Fetch raw news articles from all sources
        raw_articles = await news_aggregator.fetch_all_news()
        
        # 2. Process and clean the articles
        processed_articles = news_processor.process_articles(raw_articles)
        
        # 3. Store articles in database
        stored_articles = []
        for article in processed_articles:
            try:
                db_article = crud.create_article(
                    db=db,
                    title=article['title'],
                    content=article['content'],
                    url=article['url'],
                    source=article['source'],
                    image_url=article.get('image_url'),
                    published_date=datetime.fromisoformat(article['published_date'].replace('Z', '+00:00'))
                )
                stored_articles.append(db_article)
            except Exception as e:
                print(f"Error storing article: {str(e)}")
                continue
            
        return stored_articles
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add endpoint to check news sources status
@app.get("/news-sources/status")
def get_sources_status():
    return news_aggregator.get_source_status()