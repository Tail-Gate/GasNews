# GasNews — News Aggregation & Recommendation Service

GasNews is a FastAPI backend that aggregates natural gas industry news from multiple sources, normalizes articles into PostgreSQL, and serves personalized recommendations using OpenAI embeddings + cosine similarity. It also supports user accounts, bookmarks, feedback, and lightweight engagement analytics.

## Features
- **Multi-source aggregation**: fetches articles from external news APIs and standardizes fields (title/content/url/source/published date).
- **Normalized persistence**: stores articles, users, bookmarks, embeddings, and recommendation history in **PostgreSQL** via **SQLAlchemy ORM**.
- **Embeddings + similarity search**: generates embeddings for articles and returns similar articles using cosine similarity.
- **Personalized recommendations**: creates per-user recommendation feeds based on recent articles and stored similarity scores.
- **Interaction tracking**: logs user clicks/bookmarks and stores recommendation history for evaluation.
- **Authentication**: user creation/login with **bcrypt password hashing** (passlib).
- **Operational endpoints**: health checks, debug endpoints, and embedding coverage/progress tracking.

## Tech Stack
- **Backend**: FastAPI (Python)
- **DB/ORM**: PostgreSQL, SQLAlchemy
- **ML**: OpenAI embeddings, cosine similarity
- **Auth**: passlib (bcrypt)
- **Observability**: Python logging
- **APIs**: JSON REST endpoints

## Data Model (high level)
- `Article`: normalized news articles
- `User`: user accounts
- `Bookmark`: user→article saves
- `ArticleEmbedding`: embedding vector + model metadata per article
- `RecommendationHistory`: recommended items + similarity score + interaction + feedback

## API Overview
Core routes:
- `GET /health` — health check  
- `
