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
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta
from sqlalchemy import func, case  
from passlib.context import CryptContext


# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Initialize news sources
news_sources = [
    NewsAPISource(os.getenv("NEWS_API_KEY")),
    NewsDataSource(os.getenv("NEWSCATCHER_API_KEY")),
    CurrentsSource(os.getenv("CURRENTS_API_KEY")),
    GNewsSource(os.getenv("GNEWS_API_KEY")),
    NewsData(os.getenv("NEWSDATA_API_KEY"))
]

# Initialize services
embeddings_service = EmbeddingsService(api_key=os.getenv("OPENAI_API_KEY"))

news_aggregator = NewsAggregator(news_sources)
news_processor = NewsProcessor()
news_notifier = NewsNotifier()

#Creating FastAPI
app = FastAPI()

# Create database tables
models.Base.metadata.create_all(bind=engine)



# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Creates password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Initialize router
router = APIRouter(prefix="/recommendations", tags=["recommendations"])

@router.get("/debug/similar/{article_id}")
async def debug_similar_articles(
    article_id: int,
    threshold: float = 0.5,
    limit: int = 5,
    db: Session = Depends(get_db)
):
    """Debug endpoint for similarity search"""
    try:
        # Get source article
        article = db.query(models.Article).filter(models.Article.id == article_id).first()
        if not article:
            return {"error": "Article not found"}

        # Get article embedding
        source_embedding = db.query(models.ArticleEmbedding).filter(
            models.ArticleEmbedding.article_id == article_id
        ).first()
        
        if not source_embedding:
            return {
                "error": "No embedding found for article",
                "article_id": article_id
            }

        # Get all other articles with embeddings
        embeddings = db.query(models.ArticleEmbedding, models.Article).join(
            models.Article
        ).filter(
            models.Article.id != article_id
        ).all()

        # Calculate similarities
        similarities = []
        for embedding, target_article in embeddings:
            similarity = embeddings_service.compute_similarity(
                source_embedding.embedding_vector,
                embedding.embedding_vector
            )
            similarities.append({
                "article_id": target_article.id,
                "title": target_article.title,
                "similarity": similarity,
                "source": target_article.source,
                "published_date": target_article.published_date.isoformat()
            })

        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "source_article": {
                "id": article.id,
                "title": article.title,
                "has_embedding": True
            },
            "total_comparisons": len(similarities),
            "threshold": threshold,
            "similar_articles": similarities[:limit],
            "similarity_range": {
                "max": max(s["similarity"] for s in similarities) if similarities else 0,
                "min": min(s["similarity"] for s in similarities) if similarities else 0,
                "avg": sum(s["similarity"] for s in similarities) / len(similarities) if similarities else 0
            }
        }
    except Exception as e:
        return {"error": str(e)}

@router.get("/debug/user/{user_id}")
async def debug_user_setup(
    user_id: int,
    db: Session = Depends(get_db)
):
    """Debug endpoint for user recommendation setup"""
    try:
        # Check user exists
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if not user:
            return {"error": "User not found"}

        # Get user's bookmarks
        bookmarks = db.query(models.Bookmark).filter(
            models.Bookmark.user_id == user_id
        ).all()

        # Get recommendation history
        recommendations = db.query(models.RecommendationHistory).filter(
            models.RecommendationHistory.user_id == user_id
        ).all()

        return {
            "user_exists": True,
            "bookmarks_count": len(bookmarks),
            "recommendations_count": len(recommendations),
            "recent_bookmarks": [
                {
                    "article_id": b.article_id,
                    "created_at": b.created_at.isoformat()
                } for b in bookmarks[-5:]  # Last 5 bookmarks
            ] if bookmarks else [],
            "recent_recommendations": [
                {
                    "article_id": r.recommended_article_id,
                    "similarity_score": r.similarity_score,
                    "created_at": r.created_at.isoformat()
                } for r in recommendations[-5:]  # Last 5 recommendations
            ] if recommendations else []
        }
    except Exception as e:
        return {"error": str(e)}

@router.get("/debug/recommendations")
async def debug_recommendations(db: Session = Depends(get_db)):
    """Debug endpoint to check recommendation storage"""
    try:
        # Get counts
        total_recommendations = db.query(models.RecommendationHistory).count()
        
        # Get recent recommendations if any exist
        recent_recommendations = db.query(models.RecommendationHistory)\
            .order_by(models.RecommendationHistory.created_at.desc())\
            .limit(5)\
            .all()
            
        recent_details = []
        if recent_recommendations:
            for rec in recent_recommendations:
                try:
                    source_article = db.query(models.Article).filter(models.Article.id == rec.source_article_id).first()
                    recommended_article = db.query(models.Article).filter(models.Article.id == rec.recommended_article_id).first()
                    
                    recent_details.append({
                        "id": rec.id,
                        "user_id": rec.user_id,
                        "source_article": {
                            "id": source_article.id,
                            "title": source_article.title if source_article else None
                        },
                        "recommended_article": {
                            "id": recommended_article.id,
                            "title": recommended_article.title if recommended_article else None
                        },
                        "similarity_score": rec.similarity_score,
                        "created_at": rec.created_at.isoformat()
                    })
                except Exception as e:
                    recent_details.append({
                        "id": rec.id,
                        "error": str(e)
                    })
        
        return {
            "total_recommendations": total_recommendations,
            "recent_recommendations": recent_details,
            "table_info": {
                "has_recommendations": total_recommendations > 0,
                "recent_count": len(recent_details)
            }
        }
        
    except Exception as e:
        return {
            "error": f"Database inspection failed: {str(e)}"
        }

@router.post("/generate-all-embeddings")
async def generate_all_embeddings(
    batch_size: int = 10,
    db: Session = Depends(get_db)
):
    """Generate embeddings for all articles that don't have them yet"""
    try:
        # Get total count of articles needing embeddings
        need_embeddings = db.query(models.Article)\
            .outerjoin(models.ArticleEmbedding)\
            .filter(models.ArticleEmbedding.id == None)\
            .all()
        
        total_pending = len(need_embeddings)
        if total_pending == 0:
            return {
                "status": "success",
                "message": "No articles found needing embeddings"
            }
        
        # Get total article count for percentage calculation
        total_articles = db.query(models.Article).count()
        existing_embeddings = db.query(models.ArticleEmbedding).count()
        
        # Process articles
        successful = 0
        failed = 0
        errors = []
        start_time = datetime.now()
        
        for i in range(0, total_pending, batch_size):
            batch = need_embeddings[i:i + batch_size]
            batch_start_time = datetime.now()
            
            for article in batch:
                try:
                    embedding = await embeddings_service.create_embedding(db, article)
                    if embedding:
                        successful += 1
                    else:
                        failed += 1
                        errors.append({
                            "article_id": article.id,
                            "error": "Failed to generate embedding"
                        })
                except Exception as e:
                    failed += 1
                    errors.append({
                        "article_id": article.id,
                        "error": str(e)
                    })
            
            # Calculate progress and timing metrics
            processed = successful + failed
            progress = (processed / total_pending) * 100
            elapsed_time = (datetime.now() - start_time).total_seconds()
            batch_time = (datetime.now() - batch_start_time).total_seconds()
            
            # Update progress in database
            progress_record = {
                "timestamp": datetime.now(),
                "total_articles": total_articles,
                "total_pending": total_pending,
                "processed": processed,
                "successful": successful,
                "failed": failed,
                "progress_percentage": progress,
                "elapsed_time": elapsed_time,
                "batch_time": batch_time,
                "errors": errors[-10:]  # Keep last 10 errors
            }
            
            # Store progress in memory (or database if you prefer)
            app.state.embedding_progress = progress_record
            
        return {
            "status": "completed",
            "total_articles": total_articles,
            "total_processed": successful + failed,
            "successful": successful,
            "failed": failed,
            "completion_time": elapsed_time,
            "average_time_per_article": elapsed_time / (successful + failed) if (successful + failed) > 0 else 0,
            "errors": errors[:10] if errors else []
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@router.get("/embedding-progress")
async def get_embedding_progress(db: Session = Depends(get_db)):
    """Get current progress of embedding generation"""
    try:
        # Get current counts
        total_articles = db.query(models.Article).count()
        total_embeddings = db.query(models.ArticleEmbedding).count()
        
        # Get progress from state if available
        progress = getattr(app.state, 'embedding_progress', None)
        
        if progress:
            return {
                "status": "in_progress",
                "total_articles": total_articles,
                "current_embeddings": total_embeddings,
                "coverage_percentage": (total_embeddings / total_articles * 100) if total_articles > 0 else 0,
                "detailed_progress": progress
            }
        else:
            return {
                "status": "idle",
                "total_articles": total_articles,
                "current_embeddings": total_embeddings,
                "coverage_percentage": (total_embeddings / total_articles * 100) if total_articles > 0 else 0
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

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
        
        # Log article details    
        logger.info(f"Testing embedding for article: {article_id}")
        logger.info(f"Article title: {article.title}")
        logger.info(f"Content length: {len(article.content) if article.content else 0}")
            
        # 2. Generate embedding
        start_time = datetime.now()
        try:
            embedding = await embeddings_service.create_embedding(db, article)
        except Exception as e:
            logger.error(f"Embedding generation error: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to generate embedding: {str(e)}",
                "article_id": article_id,
                "article_details": {
                    "title": article.title,
                    "content_length": len(article.content) if article.content else 0
                }
            }
            
        generation_time = (datetime.now() - start_time).total_seconds()
        
        if not embedding:
            return {
                "status": "error",
                "message": "Failed to generate embedding - no error details available",
                "article_id": article_id
            }
            
        return {
            "status": "success",
            "article": {
                "id": article.id,
                "title": article.title,
                "content_length": len(article.content) if article.content else 0
            },
            "embedding_stats": {
                "vector_length": len(embedding.embedding_vector),
                "generation_time_seconds": generation_time,
                "model_version": embedding.model_version
            }
        }
    except Exception as e:
        logger.error(f"Test pipeline error: {str(e)}")
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

@router.post("/generate-initial-recommendations/{user_id}")
async def generate_initial_recommendations(
    user_id: int,
    db: Session = Depends(get_db)
):
    """Generate initial set of recommendations for a user"""
    try:
        # Get recent articles
        recent_articles = db.query(models.Article)\
            .order_by(models.Article.published_date.desc())\
            .limit(10)\
            .all()

        # Debug info
        process_info = {
            "recent_articles_count": len(recent_articles),
            "similar_articles_found": 0,
            "recommendations_created": 0
        }

        recommendations = []
        for article in recent_articles:
            # Get similar articles
            similar_articles = await embeddings_service.get_similar_articles(
                db, article, limit=3
            )
            
            process_info["similar_articles_found"] += len(similar_articles)
            
            # Store recommendations
            for similar_article, similarity_score in similar_articles:
                if similarity_score > 0.4:  # Add minimum similarity threshold
                    recommendation = models.RecommendationHistory(
                        source_article_id=article.id,
                        recommended_article_id=similar_article.id,
                        user_id=user_id,
                        similarity_score=similarity_score,
                        recommendation_type="topic",
                        features_used={"embedding_similarity": similarity_score}
                    )
                    recommendations.append(recommendation)
                    process_info["recommendations_created"] += 1
        
        # Batch insert recommendations
        if recommendations:
            db.bulk_save_objects(recommendations)
            db.commit()
        
        return {
            "status": "success",
            "process_info": process_info,
            "recommendations_generated": len(recommendations)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "process_info": process_info if 'process_info' in locals() else None
        }
    
# CORS middleware and router
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


# App endpoints
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
    # Check if user already exists
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash the password
    hashed_password = pwd_context.hash(user.password)
    
    # Create user with hashed password
    db_user = crud.create_user(
        db=db, 
        username=user.username, 
        email=user.email, 
        hashed_password=hashed_password
    )
    return db_user

@app.get("/users/{user_id}")
def read_user(user_id: int, db: Session = Depends(get_db)):
    user = crud.get_user(db, user_id=user_id)
    print(f"User fetched: {user}")
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


