# Docker Compose Demo

## Overview

This is a docker compose demo app to show how multiple services can be run together.

- A Redis instance for caching
- A Postgres database to store vectorized data and requests
- A FastAPI python app
- An LLM model via Docker Model Runner

## Quick Start

- Ensure Docker Desktop is installed and up to latest version for Docker Model Runner.
- Open a terminal and `cd` into this directory.
- Run `docker compose up --watch`. This may take a few minutes to download images on the first run.
- Go to `http://localhost:9494/docs` in your browser
