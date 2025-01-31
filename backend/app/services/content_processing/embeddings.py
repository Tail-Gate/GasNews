from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
import logging
from sqlalchemy.orm import Session
from . import models
import json
import asyncio
from functools import lru_cache
from openai import OpenAI

class EmbeddingsService:
    def __init__(self, api_key: str):
        # Initialize OpenAI client with DeepSeek configuration
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"  # Using v1 endpoint for OpenAI compatibility
        )
        self.model_version = "deepseek-chat"  # Using DeepSeek-V3
        self.logger = logging.getLogger(__name__)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding using DeepSeek's OpenAI-compatible API"""
        try:
            response = await self.client.embeddings.create(
                model=self.model_version,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Error getting embedding: {str(e)}")
            raise


    def _prepare_article_text(self, article: models.Article) -> str:
        """Prepare article text for embedding"""
        return f"{article.title}\n\n{article.content}"

    async def create_embedding(self, db: Session, article: models.Article) -> Optional[models.ArticleEmbedding]:
        """Create and store embedding for an article"""
        try:
            # Check if embedding already exists
            existing = db.query(models.ArticleEmbedding).filter(
                models.ArticleEmbedding.article_id == article.id
            ).first()
            
            if existing:
                return existing

            # Get embedding
            text = self._prepare_article_text(article)
            embedding_vector = await self._get_embedding(text)

            # Create embedding record
            embedding = models.ArticleEmbedding(
                article_id=article.id,
                embedding_vector=embedding_vector,
                model_version=self.model_version
            )
            
            db.add(embedding)
            db.commit()
            db.refresh(embedding)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error creating embedding for article {article.id}: {str(e)}")
            db.rollback()
            return None

    def compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    async def get_similar_articles(
        self,
        db: Session,
        article: models.Article,
        limit: int = 10,
        min_age_days: int = 0,
        max_age_days: int = 45  # 1.5 months
    ) -> List[Tuple[models.Article, float]]:
        """Get similar articles based on embedding similarity"""
        try:
            # Get or create embedding for source article
            source_embedding = await self.create_embedding(db, article)
            if not source_embedding:
                return []

            # Calculate date range
            now = datetime.now(datetime.UTC)
            min_date = now - timedelta(days=max_age_days)
            max_date = now - timedelta(days=min_age_days)

            # Get all article embeddings within date range
            embeddings = db.query(models.ArticleEmbedding, models.Article).join(
                models.Article
            ).filter(
                models.Article.published_date >= min_date,
                models.Article.published_date <= max_date,
                models.Article.id != article.id
            ).all()

            # Calculate similarities
            similarities = []
            for embedding, target_article in embeddings:
                similarity = self.compute_similarity(
                    source_embedding.embedding_vector,
                    embedding.embedding_vector
                )
                similarities.append((target_article, similarity))

            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:limit]

        except Exception as e:
            self.logger.error(f"Error finding similar articles: {str(e)}")
            return []

    async def refresh_recommendations(
        self,
        db: Session,
        user_id: int,
        topic_ratio: int = 2,
        style_ratio: int = 1
    ) -> bool:
        """Refresh recommendations for a user"""
        try:
            # Create new batch record
            batch = models.RecommendationBatch(
                user_id=user_id,
                status="processing",
                next_run_at=datetime.now(datetime.UTC) + timedelta(hours=12)
            )
            db.add(batch)
            db.commit()

            try:
                # Get user's recent interactions
                recent_interactions = db.query(models.Article).join(
                    models.Bookmark
                ).filter(
                    models.Bookmark.user_id == user_id
                ).order_by(
                    models.Bookmark.created_at.desc()
                ).limit(5).all()

                all_recommendations = []
                for article in recent_interactions:
                    # Get topic-similar articles
                    topic_similar = await self.get_similar_articles(
                        db, article, limit=topic_ratio
                    )
                    
                    # Store recommendations
                    for similar_article, similarity in topic_similar:
                        recommendation = models.RecommendationHistory(
                            source_article_id=article.id,
                            recommended_article_id=similar_article.id,
                            user_id=user_id,
                            similarity_score=similarity,
                            recommendation_type="topic",
                            features_used=json.dumps({
                                "embedding_similarity": similarity,
                                "article_age_days": (datetime.now(datetime.UTC) - similar_article.published_date).days
                            })
                        )
                        all_recommendations.append(recommendation)

                # Batch insert recommendations
                db.bulk_save_objects(all_recommendations)
                
                # Update batch status
                batch.status = "completed"
                batch.metrics = json.dumps({
                    "recommendations_generated": len(all_recommendations),
                    "source_articles": len(recent_interactions)
                })
                
                db.commit()
                return True

            except Exception as e:
                batch.status = "failed"
                batch.error_message = str(e)
                db.commit()
                raise

        except Exception as e:
            self.logger.error(f"Error refreshing recommendations: {str(e)}")
            return False