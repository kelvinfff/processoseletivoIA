import shutil
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Projeto 3 — Otimização do Modelo (Exportação para Edge)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.pt"
#   2. Exportar para TensorFlow Lite via model.export(format="tflite")
#      (a Ultralytics gera automaticamente "model.tflite" na mesma pasta)
# ---------------------------------------------------------------------------

model = YOLO("model.pt")
model.export(format="tflite", imgsz=640, quantize=8, data="dataset/data.yaml")
shutil.copy("model_int8.tflite", "model.tflite")