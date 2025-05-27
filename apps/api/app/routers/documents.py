from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Path, status
from fastapi.responses import StreamingResponse
from sqlalchemy import Column, Text, select
from sqlalchemy.dialects.postgresql import UUID, BYTEA
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
import uuid
from typing import List
import os
import io
import base64
from utils.embeddings import generate_embedding_from_file
from models.document import Document


# ─── BASE MODEL ─────────────────────────────────────────────────

Base = declarative_base()

# ─── DOCUMENT MODEL ─────────────────────────────────────────────

# class Document(Base):
#     __tablename__ = "documents"

#     id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
#     name = Column(Text)
#     content_type = Column(Text)
#     content = Column(BYTEA)
#     embedding = Column(Vector(384))

# ─── CONFIG ─────────────────────────────────────────────────────

DATABASE_URL = os.getenv("POSTGRES_CONNSTR", "postgresql+asyncpg://user:password@localhost/dbname")

engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# ─── DEPENDENCY ─────────────────────────────────────────────────

async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session


# ─── ROUTER ─────────────────────────────────────────────────────

router = APIRouter(prefix="/documents", tags=["documents"])

@router.get("/", response_model=List[dict])
async def list_documents(session: AsyncSession = Depends(get_session)):
    stmt = select(Document.id, Document.name, Document.content_type)
    result = await session.execute(stmt)
    docs = result.all()

    return [
        {
            "id": str(doc.id),
            "name": doc.name,
            "content_type": doc.content_type
        }
        for doc in docs
    ]

@router.post("/")
async def upload_document(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session)
):
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Generate embedding using utility function
    try:
        embedding = generate_embedding_from_file(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {str(e)}")

    new_doc = Document(
        name=file.filename,
        content_type=file.content_type,
        content=file_bytes,
        embedding=embedding
    )

    session.add(new_doc)
    await session.commit()

    return {
        "status": "success",
        "filename": file.filename,
        "content_type": file.content_type,
        "embedding_dimension": len(embedding)
    }

# ─── GET: Document by ID ────────────────────────────────────────

@router.get("/{document_id}", response_model=dict)
async def get_document_by_id(
    document_id: uuid.UUID = Path(..., description="The UUID of the document"),
    session: AsyncSession = Depends(get_session)
):
    stmt = select(Document).where(Document.id == document_id)
    result = await session.execute(stmt)
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Encode binary content to base64 string for JSON-safe response
    content_base64 = base64.b64encode(doc.content).decode("utf-8") if doc.content else None

    return {
        "id": str(doc.id),
        "name": doc.name,
        "content_type": doc.content_type,
        "content": content_base64,
        "embedding": [float(x) for x in doc.embedding] if doc.embedding is not None else None
    }

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID = Path(..., description="The UUID of the document"),
    session: AsyncSession = Depends(get_session)
):
    stmt = select(Document).where(Document.id == document_id)
    result = await session.execute(stmt)
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    await session.delete(doc)
    await session.commit()

    # Returning 204 No Content (empty response)
    return


@router.get("/{document_id}/download")
async def download_document(
    document_id: uuid.UUID = Path(..., description="The UUID of the document"),
    session: AsyncSession = Depends(get_session)
):
    stmt = select(Document).where(Document.id == document_id)
    result = await session.execute(stmt)
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    if not doc.content:
        raise HTTPException(status_code=404, detail="No content available for this document.")

    file_like = io.BytesIO(doc.content)

    return StreamingResponse(
        file_like,
        media_type=doc.content_type or "application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{doc.name}"'
        }
    )