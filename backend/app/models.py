from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Array, Float, JSON
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime, UTC
import numpy as np

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    email = Column(String, unique=True, index=True)
    
    bookmarks = relationship("Bookmark", back_populates="user")

class Article(Base):
    __tablename__ = "articles"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    content = Column(String)
    url = Column(String, unique=True)
    source = Column(String)
    published_date = Column(DateTime, default=lambda: datetime.now(UTC))
    image_url = Column(String, nullable=True)

class Bookmark(Base):
    __tablename__ = "bookmarks"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    article_id = Column(Integer, ForeignKey("articles.id"))
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    
    # Relationships
    user = relationship("User", back_populates="bookmarks")
    article = relationship("Article")

class ArticleEmbedding(Base):
    __tablename__ = "article_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("articles.id"), unique=True)
    embedding_vector = Column(Array(Float))  # Store embedding as array of floats
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    last_updated = Column(DateTime, default=lambda: datetime.now(UTC))
    model_version = Column(String)  # Track which model version generated the embedding
    
    # Relationship
    article = relationship("Article", back_populates="embedding")

class RecommendationHistory(Base):
    __tablename__ = "recommendation_history"
    
    id = Column(Integer, primary_key=True, index=True)
    source_article_id = Column(Integer, ForeignKey("articles.id"))
    recommended_article_id = Column(Integer, ForeignKey("articles.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    similarity_score = Column(Float)
    recommendation_type = Column(String)  # 'topic' or 'style'
    features_used = Column(JSON)  # Store which features led to this recommendation
    was_clicked = Column(Boolean, default=False)
    was_bookmarked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    
    # Relationships
    source_article = relationship("Article", foreign_keys=[source_article_id])
    recommended_article = relationship("Article", foreign_keys=[recommended_article_id])
    user = relationship("User")

class RecommendationBatch(Base):
    __tablename__ = "recommendation_batches"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    next_run_at = Column(DateTime)
    status = Column(String)  # 'pending', 'processing', 'completed', 'failed'
    error_message = Column(String, nullable=True)
    metrics = Column(JSON, nullable=True)  # Store performance metrics
    
    # Relationship
    user = relationship("User")

# Add relationship to Article model
Article.embedding = relationship("ArticleEmbedding", back_populates="article", uselist=False)