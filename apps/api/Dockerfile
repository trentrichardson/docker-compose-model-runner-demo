FROM python:3.12

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install -r /code/requirements.txt

COPY ./app /code/app

EXPOSE 9494

CMD ["FASTAPI", "run", "/code/app/main.py", "--port", "9494"]