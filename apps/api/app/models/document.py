import uuid
from sqlalchemy import Column, String, LargeBinary, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
from sqlalchemy.types import TIMESTAMP
from sqlalchemy.dialects.postgresql import BYTEA

from pgvector.sqlalchemy import Vector  # if you're using pgvector-sqlalchemy

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text)
    content_type = Column(Text)
    content = Column(BYTEA)
    embedding = Column(Vector(384))
