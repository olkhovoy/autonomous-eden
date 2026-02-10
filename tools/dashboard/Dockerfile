FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    nicegui>=1.4 \
    requests \
    uvicorn

COPY tools/dashboard/app.py /app/app.py

EXPOSE 8110

CMD ["python", "app.py"]
