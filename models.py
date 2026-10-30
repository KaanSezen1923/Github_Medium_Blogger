from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, ARRAY
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # İlişkiler
    repositories = relationship("Repository", back_populates="owner_user", cascade="all, delete-orphan")

class Repository(Base):
    __tablename__ = "repositories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), ForeignKey("users.user_id", ondelete="CASCADE"))
    owner = Column(String(255), nullable=False)
    repo_name = Column(String(255), nullable=False)
    repo_url = Column(Text, nullable=False)
    status = Column(String(50), default="pending")
    focus_area = Column(Text, nullable=True)
    audience = Column(String(255), nullable=True)
    tone = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # İlişkiler
    owner_user = relationship("User", back_populates="repositories")
    blog_output = relationship("BlogOutput", back_populates="repository", uselist=False, cascade="all, delete-orphan")

class BlogOutput(Base):
    __tablename__ = "blog_outputs"

    id = Column(Integer, primary_key=True, index=True)
    repo_id = Column(Integer, ForeignKey("repositories.id", ondelete="CASCADE"))
    analysis_data = Column(JSONB, nullable=True)
    outline_data = Column(JSONB, nullable=True)
    markdown_content = Column(Text, nullable=True)
    warnings = Column(ARRAY(Text), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # İlişkiler
    repository = relationship("Repository", back_populates="blog_output")