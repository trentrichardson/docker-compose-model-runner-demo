# Docker Compose Demo

## Overview

This is a docker compose demo app to show how multiple services can be run together.

- A Redis instance for caching
- A Postgres database to store vectorized data and requests
- A FastAPI python app
- An LLM model via Docker Model Runner

## Quick Start

- Ensure Docker Desktop is installed and up to latest version for Docker Model Runner.
- Make a copy of `bin/lcl/.env-template` to `.env` and set the env variables to your preference.
- Open a terminal and `cd` into `bin/lcl`.
- Run `docker compose up --watch` or `docker-compose up --watch` depending on your system. 
  - This may take a few minutes to download images on the first run.
- Go to `http://localhost:9494/docs` in your browser

## About

- Github: https://github.com/trentrichardson/docker-compose-model-runner-demo
- YouTube: https://youtu.be/irySzZZL8dU?si=dIstAwcBtg2uFh4f
- Author: https://trentrichardson.com/
