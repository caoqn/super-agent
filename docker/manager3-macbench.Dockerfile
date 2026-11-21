FROM python:3.11-slim

WORKDIR /app
COPY requirements/manager3-macbench.txt requirements/manager3-macbench.txt
RUN pip install --no-cache-dir -r requirements/manager3-macbench.txt
COPY methods methods
COPY benchmarks benchmarks
COPY configs configs
COPY scripts scripts
RUN mkdir -p /app/results/manager3/macbench

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/mnt/macbench/src
ENV MAC_BENCH_ROOT=/mnt/macbench
ENV MAC_BENCH_SPLIT=test
ENTRYPOINT ["python", "-m", "methods.manager3.macbench.runner"]
