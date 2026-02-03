# Dockerfile para entrenamiento YOLO con TensorBoard garantizado
FROM ultralytics/ultralytics:latest

# Instalar dependencias adicionales
RUN pip install --no-cache-dir \
    transformers \
    supervision \
    groundingdino-py \
    tensorboard>=2.14.0 \
    torch-tb-profiler \
    python-dotenv

# Variables de entorno
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4192
ENV HOME=/tmp

# Crear directorio de configuración de Ultralytics
RUN mkdir -p /root/.config/Ultralytics && \
    echo "settings_version: 0.0.6" > /root/.config/Ultralytics/settings.yaml && \
    echo "tensorboard: true" >> /root/.config/Ultralytics/settings.yaml && \
    echo "runs_dir: /workspace/runs" >> /root/.config/Ultralytics/settings.yaml

# Crear directorios de trabajo
RUN mkdir -p /workspace/runs /workspace/scripts /workspace/datasets

WORKDIR /workspace
EXPOSE 6006

CMD ["/bin/bash"]