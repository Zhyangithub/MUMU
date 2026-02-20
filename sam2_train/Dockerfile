FROM --platform=linux/amd64 pytorch/pytorch

ENV PYTHONUNBUFFERED=1

# Install system dependencies for OpenCV and other potential needs
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/resources
COPY sam2_train/ /opt/app/sam2_train/
RUN rm -f /opt/app/sam2_train/_C.so && find /opt/app -name "*.pyc" -delete
# Install requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user model.py /opt/app/
COPY --chown=user:user cfg.py /opt/app/
COPY --chown=user:user sam2_train /opt/app/sam2_train
RUN rm -f /opt/app/sam2_train/_C.so && find /opt/app -name "*.pyc" -delete
COPY --chown=user:user conf /opt/app/conf

# 注意：模型权重通过 Algorithm Models 上传到 /opt/ml/model/
# 不再需要 COPY checkpoints

ENTRYPOINT ["python", "inference.py"]
