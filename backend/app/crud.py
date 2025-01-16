from sqlalchemy.orm import Session
from datetime import datetime
from . import models, schemas
from typing import List, Optional
from fastapi import HTTPException

# Article CRUD operations
def create_article(
    db: Session,
    title: str,
    content: str,
    url: str,
    source: str,
    image_url: Optional[str] = None,
    published_date: Optional[datetime] = None
) -> models.Article:
    # First check if article with same URL exists
    existing_article = db.query(models.Article).filter(models.Article.url == url).first()
    if existing_article:
        # If article exists, return it without creating a new one
        return existing_article
        
    # If article doesn't exist, create new one
    db_article = models.Article(
        title=title,
        content=content,
        url=url,
        source=source,
        image_url=image_url,
        published_date=published_date or datetime.utcnow()
    )
    
    db.add(db_article)
    db.commit()
    db.refresh(db_article)
    return db_article

def get_article(db: Session, article_id: int) -> Optional[models.Article]:
    article = db.query(models.Article).filter(models.Article.id == article_id).first()
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found")
    return article

def get_articles(
    db: Session, 
    skip: int = 0, 
    limit: int = 100
) -> List[models.Article]:
    return db.query(models.Article)\
        .order_by(models.Article.published_date.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()

def get_article_by_url(db: Session, url: str) -> Optional[models.Article]:
    return db.query(models.Article).filter(models.Article.url == url).first()

# User CRUD operations
def create_user(db: Session, username: str, email: str, hashed_password: str) -> models.User:
    db_user = models.User(
        username=username,
        email=email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user(db: Session, user_id: int) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_username(db: Session, username: str) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.username == username).first()

def get_user_by_email(db: Session, email: str) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.email == email).first()

# Bookmark CRUD operations
def create_bookmark(
    db: Session, 
    user_id: int, 
    article_id: int
) -> Optional[models.Bookmark]:
    # Check if bookmark already exists
    existing_bookmark = db.query(models.Bookmark).filter(
        models.Bookmark.user_id == user_id,
        models.Bookmark.article_id == article_id
    ).first()
    
    if existing_bookmark:
        return existing_bookmark
        
    db_bookmark = models.Bookmark(
        user_id=user_id,
        article_id=article_id,
        created_at=datetime.utcnow()
    )
    db.add(db_bookmark)
    db.commit()
    db.refresh(db_bookmark)
    return db_bookmark

def get_user_bookmarks(db: Session, user_id: int) -> List[models.Bookmark]:
    return db.query(models.Bookmark)\
        .filter(models.Bookmark.user_id == user_id)\
        .order_by(models.Bookmark.created_at.desc())\
        .all()

def delete_bookmark(db: Session, user_id: int, article_id: int) -> bool:
    bookmark = db.query(models.Bookmark).filter(
        models.Bookmark.user_id == user_id,
        models.Bookmark.article_id == article_id
    ).first()
    
    if bookmark:
        db.delete(bookmark)
        db.commit()
        return True
    return False