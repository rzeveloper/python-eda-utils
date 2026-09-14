# preprocessing.py
# Funciones reutilizables de limpieza y transformación de datos.
# Importar desde los notebooks con: from src.preprocessing import ...

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from typing import Sequence

def get_numeric_columns(df: pd.DataFrame) -> list[str]:
  """Obtiene los nombres de las columnas numéricas del DataFrame.

  Parameters:
    df: DataFrame de entrada.

  Returns:
    Lista con nombres de columnas numéricas.
  """
  return df.select_dtypes(include=['number']).columns.tolist()

def get_categorical_columns(df: pd.DataFrame) -> list[str]:
  """Obtiene los nombres de las columnas categóricas del DataFrame.

  Parameters:
    df: DataFrame de entrada.

  Returns:
    Lista con nombres de columnas categóricas.
  """
  return df.select_dtypes(include=['object', 'category']).columns.tolist()

def plot_numeric_histograms(
  df: pd.DataFrame,
  numeric_columns: Sequence[str],
  fig_per_row: int = 4
) -> None:
  """Grafica histogramas para un conjunto de columnas numéricas.

  Parameters:
    df: DataFrame de entrada.
    numeric_columns: Secuencia con nombres de columnas numéricas a graficar.
    fig_per_row: Cantidad de gráficos por fila.
  """
  n_cols = fig_per_row  # histogramas por fila
  n_rows = math.ceil(len(numeric_columns) / n_cols)
  size = (5 * n_cols, 4 * n_rows) if fig_per_row > 1 else (10, 8)

  fig, axes = plt.subplots(n_rows, n_cols, figsize=size)
  axes = np.atleast_1d(axes).flatten()

  for ax, col in zip(axes, numeric_columns):
    sns.histplot(df[col], bins=30, kde=True, ax=ax)
    ax.set_title('')
    ax.set_xlabel(col)
    ax.set_ylabel('Frecuencia')

  # Ocultar ejes vacíos
  for ax in axes[len(numeric_columns):]:
    ax.set_visible(False)

  plt.tight_layout()
  plt.show()

def plot_numeric_boxplots(
  df: pd.DataFrame,
  numeric_columns: Sequence[str],
  fig_per_row: int = 4
) -> None:
  """Grafica diagramas de caja para un conjunto de columnas numéricas.

  Parameters:
    df: DataFrame de entrada.
    numeric_columns: Secuencia con nombres de columnas numéricas a graficar.
    fig_per_row: Cantidad de gráficos por fila.
  """
  n_cols = fig_per_row  # diagramas por fila
  n_rows = math.ceil(len(numeric_columns) / n_cols)
  size = (5 * n_cols, 4 * n_rows) if fig_per_row > 1 else (10, 8)

  fig, axes = plt.subplots(n_rows, n_cols, figsize=size)
  axes = np.atleast_1d(axes).flatten()

  for ax, col in zip(axes, numeric_columns):
    sns.boxplot(x=df[col], ax=ax)
    ax.set_title('')
    ax.set_xlabel(col)

  # Ocultar ejes vacíos
  for ax in axes[len(numeric_columns):]:
    ax.set_visible(False)

  plt.tight_layout()
  plt.show()

def plot_categorical_countplots(
  df: pd.DataFrame,
  categorical_columns: Sequence[str],
  fig_per_row: int = 4,
  top_n: int = 15
) -> None:
  """Grafica countplots para un conjunto de columnas categóricas.

  Parameters:
    df: DataFrame de entrada.
    categorical_columns: Secuencia con nombres de columnas categóricas a graficar.
    fig_per_row: Cantidad de gráficos por fila.
    top_n: Número máximo de categorías a mostrar en cada gráfico.
  """
  n_cols = fig_per_row  # gráficos por fila
  n_rows = math.ceil(len(categorical_columns) / n_cols)
  size = (6.5 * n_cols, 4 * n_rows) if fig_per_row > 1 else (10, 8)

  fig, axes = plt.subplots(n_rows, n_cols, figsize=size)
  axes = np.atleast_1d(axes).flatten()

  for ax, col in zip(axes, categorical_columns):
    top_categories = df[col].value_counts().nlargest(top_n).index

    x_label = f"{col} - ({top_n} de {df[col].nunique(dropna=True)})"
    if df[col].nunique(dropna=True) <= top_n:
      x_label = f"{col}"

    sns.countplot(x=df[col], order=top_categories, ax=ax)
    ax.set_title('')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Frecuencia')
    plt.setp(ax.get_xticklabels(), rotation=90)

  # Ocultar ejes vacíos
  for ax in axes[len(categorical_columns):]:
    ax.set_visible(False)

  plt.tight_layout()
  plt.show()

def plot_numeric_boxplots_by_target(
  df: pd.DataFrame,
  target: str,
  numeric_columns: Sequence[str],
  fig_per_row: int = 4
) -> None:
  """Grafica boxplots de columnas numéricas separados por una variable objetivo.

  Parameters:
    df: DataFrame de entrada.
    target: Nombre de la columna objetivo para agrupar.
    numeric_columns: Secuencia con nombres de columnas numéricas a graficar.
    fig_per_row: Cantidad de gráficos por fila.
  """
  n_cols = fig_per_row  # diagramas por fila
  n_rows = math.ceil(len(numeric_columns) / n_cols)
  size = (5 * n_cols, 4 * n_rows) if fig_per_row > 1 else (10, 8)

  fig, axes = plt.subplots(n_rows, n_cols, figsize=size)
  axes = np.atleast_1d(axes).flatten()

  for ax, col in zip(axes, numeric_columns):
    sns.boxplot(data=df, x=target, y=col, ax=ax)
    ax.set_title('')
    ax.set_xlabel(target)
    ax.set_ylabel(col)

  # Ocultar ejes vacíos
  for ax in axes[len(numeric_columns):]:
    ax.set_visible(False)

  plt.tight_layout()
  plt.show()

def get_boundaries_iqr(series: pd.Series) -> tuple[float, float]:
  """Calcula los límites inferior y superior para detectar outliers usando el método IQR.

  Parameters:
    series: Serie numérica para calcular los límites.

  Returns:
    Tupla con (límite inferior, límite superior).
  """
  Q1 = series.quantile(0.25)
  Q3 = series.quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  return lower_bound, upper_bound