FROM python:3.11-slim
WORKDIR /app
RUN pip install mcp aiohttp
COPY process_all_tickets.py .
CMD ["python", "process_all_tickets.py"]