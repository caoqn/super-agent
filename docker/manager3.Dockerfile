FROM python:3.11-slim

WORKDIR /app
COPY requirements/manager3-masbench.txt requirements/manager3-masbench.txt
RUN pip install --no-cache-dir -r requirements/manager3-masbench.txt
COPY methods methods
COPY benchmarks benchmarks
RUN mkdir -p /app/results/manager3/masbench/test90

ENV PYTHONUNBUFFERED=1
ENV MASBENCH_ROOT=/mnt/masbench
ENV MASBENCH_OUT_DIR=/app/results/manager3/masbench/test90
ENTRYPOINT ["python", "-m", "methods.manager3.masbench.runner"]
