# evaluation.py
# Funciones reutilizables de evaluación y visualización de métricas.

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Sequence

def plot_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> tuple[int, int, int, int]:
  """Grafica la matriz de confusión binaria y retorna TN, FP, FN y TP.

  Args:
    y_true: Etiquetas reales.
    y_pred: Etiquetas predichas por el modelo.

  Returns:
    Tupla con el orden (tn, fp, fn, tp).
  """
  matrix = confusion_matrix(y_true, y_pred)

  # Orden estándar de ravel para matriz binaria: TN, FP, FN, TP.
  tn, fp, fn, tp = matrix.ravel()

  display = ConfusionMatrixDisplay(confusion_matrix=matrix)
  display.plot()
  plt.show()

  return (tn, fp, fn, tp)