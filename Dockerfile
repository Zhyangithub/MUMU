FROM --platform=linux/amd64 pytorch/pytorch

# 1. Basic environment
ENV PYTHONUNBUFFERED=1

# System dependencies (needed for opencv-python and sam2)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user
WORKDIR /opt/app

# 2. Copy requirements and sam2_train first (for pip install layer caching)
COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user sam2_train/ /opt/app/sam2_train/

# Remove local compiled extension/cache artifacts to avoid ABI issues in GC runtime.
RUN rm -f /opt/app/sam2_train/_C.so && find /opt/app -name "*.pyc" -delete

# 3. Install Python dependencies
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# 4. Copy application code
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user model.py /opt/app/
COPY --chown=user:user cfg.py /opt/app/

# 5. Copy resource/config directories (must exist, even if empty)
COPY --chown=user:user resources/ /opt/app/resources/

# Model weights: uploaded via Algorithm Models -> /opt/ml/model/

ENTRYPOINT ["python", "inference.py"]
