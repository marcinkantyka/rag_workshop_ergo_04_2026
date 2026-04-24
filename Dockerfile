FROM python:3.11-slim-bookworm

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 (promptfoo requires ^20.20.0; debian bookworm ships 18)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install promptfoo — build deps needed only for better-sqlite3 compilation,
# purged immediately after to keep the image lean
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        python3-dev \
    && npm install -g promptfoo && promptfoo --version \
    && apt-get purge -y build-essential gcc g++ python3-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
# Install CPU-only torch first to avoid pulling the 2.5 GB CUDA variant
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# ── Jupyter configuration ──────────────────────────────────────────────────
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" \
        >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" \
        >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" \
        >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.root_dir = '/workspace'" \
        >> /root/.jupyter/jupyter_lab_config.py

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--port=8888", \
     "--no-browser", \
     "--ip=0.0.0.0"]
