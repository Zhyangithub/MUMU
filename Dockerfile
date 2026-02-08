FROM --platform=linux/amd64 pytorch/pytorch

ENV PYTHONUNBUFFERED=1

# Install system dependencies for OpenCV and other potential needs
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

# ---- Install Python dependencies first (for Docker layer caching) ----
COPY --chown=user:user requirements.txt /opt/app/

# sam2_train must be present before pip install in case it's needed
# as a local dependency (e.g. imported during setup)
COPY --chown=user:user sam2_train /opt/app/sam2_train

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# ---- Copy application code ----
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user model.py /opt/app/
COPY --chown=user:user cfg.py /opt/app/
COPY --chown=user:user resources /opt/app/resources

# Copy Hydra config directory if present (for local testing)
# On Grand Challenge this may also come from sam2_train/
COPY --chown=user:user conf /opt/app/conf

# ------------------------------------------------------------------
# Model weights:
# - For Grand Challenge: upload via "Algorithm Models" -> /opt/ml/model/
# - For local testing: place in checkpoints/ and uncomment below:
# COPY --chown=user:user checkpoints /opt/app/checkpoints
# ------------------------------------------------------------------

# Switch to non-root user (Grand Challenge requirement)
USER user

ENTRYPOINT ["python", "inference.py"]
