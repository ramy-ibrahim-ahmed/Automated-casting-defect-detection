FROM python:3.10-slim

WORKDIR /code

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /tmp/pip_*, /tmp/pip/

COPY ./src ./src

RUN useradd -m myuser
USER myuser

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
