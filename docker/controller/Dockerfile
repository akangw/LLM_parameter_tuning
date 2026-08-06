FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git openssh-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY tuning_pipeline/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-runtime.txt
COPY . /workspace

ENTRYPOINT ["/bin/bash", "/workspace/scripts/controller.sh"]
CMD ["help"]
