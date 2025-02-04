from fastapi import FastAPI, Depends, HTTPException,APIRouter
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from .database import engine, SessionLocal,get_db
from . import models, crud, schemas
from typing import List
from .services.news_sources import NewsAPISource,NewsDataSource,CurrentsSource,GNewsSource,NewsData,NewsAggregator
from .services.processor import NewsProcessor
from .services.notifier import NewsNotifier
from .services.content_processing.embeddings import EmbeddingsService
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from sqlalchemy import func, case  

# Load environment variables
load_dotenv()


router = APIRouter(prefix="/recommendations", tags=["recommendations"])

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

# Initialize router
router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# Initialize services
embeddings_service = EmbeddingsService(api_key=os.getenv("DEEPSEEK_API_KEY"))

@router.get("/test-embedding/{article_id}")
async def test_embedding_pipeline(
    article_id: int,
    db: Session = Depends(get_db)
):
    """Test the complete embedding pipeline for an article"""
    try:
        # 1. Get the article
        article = db.query(models.Article).filter(models.Article.id == article_id).first()
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
            
        # 2. Generate embedding
        start_time = datetime.now()
        embedding = await embeddings_service.create_embedding(db, article)
        generation_time = (datetime.now() - start_time).total_seconds()
        
        if not embedding:
            return {
                "status": "error",
                "message": "Failed to generate embedding",
                "article_id": article_id
            }
            
        # 3. Get similar articles
        similar_articles = await embeddings_service.get_similar_articles(
            db, article, limit=3
        )
        
        return {
            "status": "success",
            "article": {
                "id": article.id,
                "title": article.title,
                "source": article.source
            },
            "embedding_stats": {
                "vector_length": len(embedding.embedding_vector),
                "generation_time_seconds": generation_time,
                "model_version": embedding.model_version
            },
            "similar_articles": [
                {
                    "id": art.id,
                    "title": art.title,
                    "similarity_score": score,
                    "source": art.source
                }
                for art, score in similar_articles
            ]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "article_id": article_id
        }

@router.get("/embedding-stats")
async def get_embedding_stats(db: Session = Depends(get_db)):
    """Get statistics about embeddings in the system"""
    try:
        # Get total counts
        total_articles = db.query(models.Article).count()
        total_embeddings = db.query(models.ArticleEmbedding).count()
        
        # Get recent embeddings
        recent_embeddings = db.query(models.ArticleEmbedding)\
            .order_by(models.ArticleEmbedding.created_at.desc())\
            .limit(5)\
            .all()
            
        return {
            "total_articles": total_articles,
            "total_embeddings": total_embeddings,
            "coverage_percentage": (total_embeddings / total_articles * 100) if total_articles > 0 else 0,
            "recent_embeddings": [
                {
                    "article_id": emb.article_id,
                    "created_at": emb.created_at,
                    "model_version": emb.model_version
                }
                for emb in recent_embeddings
            ]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
@router.get("/similar/{article_id}")
async def get_similar_articles(
    article_id: int,
    limit: int = 5,
    db: Session = Depends(get_db)
):
    """Get similar articles based on embeddings"""
    article = db.query(models.Article).filter(models.Article.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    similar_articles = await embeddings_service.get_similar_articles(
        db, article, limit=limit
    )
    
    return [
        {
            "article_id": art.id,
            "title": art.title,
            "similarity_score": score,
            "url": art.url
        }
        for art, score in similar_articles
    ]

@router.post("/{article_id}/feedback")
async def submit_feedback(
    article_id: int,
    feedback: schemas.RecommendationFeedback,
    db: Session = Depends(get_db)
):
    """Submit feedback for a recommendation"""
    # Update recommendation history with feedback
    recommendation = db.query(models.RecommendationHistory).filter(
        models.RecommendationHistory.recommended_article_id == article_id,
        models.RecommendationHistory.user_id == feedback.user_id
    ).first()
    
    if recommendation:
        if feedback.feedback_type == "thumbs_up":
            recommendation.was_clicked = True
            recommendation.was_bookmarked = True
        db.commit()
    
    return {"status": "success"}

@router.get("/refresh/{user_id}")
async def refresh_user_recommendations(
    user_id: int,
    db: Session = Depends(get_db)
):
    """Manually trigger a recommendation refresh for a user"""
    success = await embeddings_service.refresh_recommendations(db, user_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to refresh recommendations")
    return {"status": "success"}


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
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

@app.get("/system/status", tags=["system"])
async def check_system_status(db: Session = Depends(get_db)):
    """Check status of the entire recommendation system using real data"""
    try:
        # Check news articles
        recent_articles = db.query(models.Article)\
            .order_by(models.Article.published_date.desc())\
            .limit(10)\
            .all()
            
        # Get source statistics
        source_counts = db.query(
            models.Article.source,
            func.count(models.Article.id).label('count')
        ).group_by(models.Article.source).all()
        
        # Check embedding tables
        embeddings_count = db.query(models.ArticleEmbedding).count()
        
        # Check recommendations
        recommendations_count = db.query(models.RecommendationHistory).count()
        
        return {
            "system_status": {
                "database_connected": True,
                "news_pipeline_active": len(recent_articles) > 0,
                "embedding_service": "not_configured",  # Will update when DeepSeek is added
                "recommendation_system": "ready_for_setup"
            },
            "data_status": {
                "total_articles": db.query(models.Article).count(),
                "recent_articles_count": len(recent_articles),
                "newest_article_date": recent_articles[0].published_date if recent_articles else None,
                "sources_breakdown": {source: count for source, count in source_counts},
                "embeddings_count": embeddings_count,
                "recommendations_count": recommendations_count
            },
            "latest_articles": [
                {
                    "id": article.id,
                    "title": article.title,
                    "source": article.source,
                    "published_date": article.published_date
                }
                for article in recent_articles[:5]  # Show last 5 articles
            ]
        }
    except Exception as e:
        return {
            "error": f"System check failed: {str(e)}",
            "database_connected": False
        }

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

@router.get("/user/{user_id}")
async def get_user_recommendations(
    user_id: int,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get personalized recommendations for a user"""
    # Get recent recommendations
    recommendations = db.query(models.RecommendationHistory).filter(
        models.RecommendationHistory.user_id == user_id,
        models.RecommendationHistory.created_at >= datetime.now() - timedelta(days=2)
    ).order_by(
        models.RecommendationHistory.created_at.desc()
    ).limit(limit).all()
    
    if not recommendations:
        return {"recommendations": [], "message": "No recent recommendations"}
    
    return {
        "recommendations": [
            {
                "article_id": rec.recommended_article_id,
                "title": rec.recommended_article.title,
                "url": rec.recommended_article.url,
                "similarity_score": rec.similarity_score,
                "source": rec.recommended_article.source,
                "published_date": rec.recommended_article.published_date
            }
            for rec in recommendations
        ]
    }

@router.post("/{article_id}/feedback")
async def submit_feedback(
    article_id: int,
    feedback: schemas.RecommendationFeedback,
    db: Session = Depends(get_db)
):
    """Submit feedback (thumbs up/down) for a recommended article"""
    # Find the most recent recommendation for this article and user
    recommendation = db.query(models.RecommendationHistory).filter(
        models.RecommendationHistory.recommended_article_id == article_id,
        models.RecommendationHistory.user_id == feedback.user_id
    ).order_by(
        models.RecommendationHistory.created_at.desc()
    ).first()
    
    if not recommendation:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    
    # Update feedback
    recommendation.was_clicked = True
    if feedback.feedback_type == "thumbs_up":
        recommendation.was_bookmarked = True
    
    # Store feedback timestamp
    recommendation.feedback_timestamp = datetime.now()
    
    db.commit()
    
    return {"status": "success"}

@router.get("/{article_id}/similar")
async def get_similar_articles(
    article_id: int,
    limit: int = 5,
    db: Session = Depends(get_db)
):
    """Get similar articles (placeholder until embeddings are implemented)"""
    # For now, return articles from the same source with similar dates
    article = db.query(models.Article).filter(models.Article.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    similar_articles = db.query(models.Article).filter(
        models.Article.source == article.source,
        models.Article.id != article_id,
        models.Article.published_date >= article.published_date - timedelta(days=7),
        models.Article.published_date <= article.published_date + timedelta(days=7)
    ).limit(limit).all()
    
    return [
        {
            "article_id": art.id,
            "title": art.title,
            "url": art.url,
            "source": art.source,
            "published_date": art.published_date
        }
        for art in similar_articles
    ]

@router.get("/stats/{user_id}")
async def get_recommendation_stats(
    user_id: int,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Get statistics about recommendations and user interaction"""
    since_date = datetime.now() - timedelta(days=days)
    
    # Get recommendation stats
    stats = db.query(
        func.count(models.RecommendationHistory.id).label('total_recommendations'),
        func.sum(case((models.RecommendationHistory.was_clicked == True, 1), else_=0)).label('clicked'),
        func.sum(case((models.RecommendationHistory.was_bookmarked == True, 1), else_=0)).label('bookmarked')
    ).filter(
        models.RecommendationHistory.user_id == user_id,
        models.RecommendationHistory.created_at >= since_date
    ).first()
    
    return {
        "period_days": days,
        "total_recommendations": stats.total_recommendations or 0,
        "interaction_rate": (stats.clicked or 0) / stats.total_recommendations if stats.total_recommendations else 0,
        "bookmark_rate": (stats.bookmarked or 0) / stats.total_recommendations if stats.total_recommendations else 0,
    }

