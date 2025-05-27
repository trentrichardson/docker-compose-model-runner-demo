CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id uuid not null default gen_random_uuid(),
    name TEXT,
    content_type TEXT,
    content bytea,
    embedding VECTOR(384),
    primary key (id)
);
CREATE INDEX IF NOT EXISTS idx_embedding ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

comment on table documents IS 'Table to hold vector embeddings of documents';
comment on column documents.name is 'Name of the document';
comment on column documents.content_type is 'Content type of the document';
comment on column documents.content is 'Content of the document';
comment on column documents.embedding is 'Vector embedding of the document';

CREATE TABLE IF NOT EXISTS conversations (
    id uuid not null default gen_random_uuid(),
    created timestamptz DEFAULT now(),
    request json,
    response json,
    response_file bytea,
    primary key (id)
);

comment on table conversations IS 'Table to hold conversations asked of documents';
comment on column conversations.created is 'Datetime the conversation was created';
comment on column conversations.request is 'JSON request to the LLM';
comment on column conversations.response is 'JSON response from the LLM';
comment on column conversations.response_file is 'File response from the LLM';