import uuid
from datetime import datetime
from sqlalchemy import Column, DateTime, JSON, LargeBinary
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base
from pydantic import BaseModel, Field
from typing import Optional


Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    created = Column(DateTime(timezone=True), default=datetime.utcnow)
    request = Column(JSON, nullable=True)
    response = Column(JSON, nullable=True)
    response_file = Column(LargeBinary, nullable=True)

class ConversationCreateRequest(BaseModel):
    message: str = Field(..., description="User input message")