from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum

# User Schemas
class UserBase(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr

class UserCreate(UserBase):
    email: str = Field(min_length=6)
    password: str = Field(min_length=8)

class User(UserBase):
    id: int
    bookmarks: List[int] = []

    class Config:
        orm_mode = True

# Article Schemas
class ArticleBase(BaseModel):
    title: str = Field(min_length=1, max_length=500)  # Increased max_length for news articles
    content: str
    url: str
    source: str
    image_url: Optional[str] = None

class ArticleCreate(ArticleBase):
    published_date: datetime = Field(default_factory=datetime.utcnow)

class Article(ArticleBase):
    id: int
    published_date: datetime

    class Config:
        orm_mode = True

# News Schemas (New)
class NewsArticleResponse(BaseModel):
    title: str
    content: str
    url: str
    source: str
    published_date: datetime
    image_url: Optional[str] = None

    class Config:
        orm_mode = True

class NewsResponse(BaseModel):
    articles: List[NewsArticleResponse]
    total_count: int
    fetch_time: datetime = Field(default_factory=datetime.utcnow)

# Bookmark Schemas
# Add this new schema for bookmark requests
class BookmarkRequest(BaseModel):
    user_id: int

class BookmarkBase(BaseModel):
    user_id: int
    article_id: int

class BookmarkCreate(BookmarkBase):
    pass

class Bookmark(BookmarkBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True

# Response Models
class UserResponse(User):
    bookmarks: List[Article] = []

class ArticleResponse(Article):
    pass

class BookmarkResponse(Bookmark):
    article: Article

class FeedbackType(str, Enum):
    thumbs_up = "thumbs_up"
    thumbs_down = "thumbs_down"

class RecommendationFeedback(BaseModel):
    user_id: int
    feedback_type: FeedbackType
    
class RecommendationResponse(BaseModel):
    article_id: int
    title: str
    similarity_score: float
    url: str