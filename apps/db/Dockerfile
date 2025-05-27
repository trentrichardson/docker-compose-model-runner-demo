FROM postgres:17.5

# Install pgvector
RUN apt-get update && apt-get install -y postgresql-server-dev-17 git build-essential \
    && git clone --branch v0.7.0 https://github.com/pgvector/pgvector.git \
    && cd pgvector && make && make install \
    && apt-get remove --purge -y git build-essential postgresql-server-dev-17 \
    && rm -rf /var/lib/apt/lists/* pgvector
