# preprocessing.py
# Funciones reutilizables de limpieza y transformación de datos.
# Importar desde los notebooks con: from src.preprocessing import ...

import math
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency


def cramers_v(x: pd.Series, y: pd.Series) -> float:
  """Calcula Cramér's V entre dos variables categóricas."""
  valid = pd.DataFrame({'x': x, 'y': y}).dropna()
  table = pd.crosstab(valid['x'], valid['y'])
  if table.empty:
    return 0.0

  chi2 = chi2_contingency(table)[0]
  observations = int(table.to_numpy().sum())
  rows, columns = table.shape
  denominator = observations * min(rows - 1, columns - 1)
  if denominator <= 0:
    return 0.0
  return float(np.sqrt(chi2 / denominator))


def get_boundaries_iqr(series: pd.Series) -> tuple[float, float]:
  """Calcula los límites inferior y superior para detectar outliers con IQR."""
  q1 = float(series.quantile(0.25))
  q3 = float(series.quantile(0.75))
  iqr = q3 - q1
  return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
  """Obtiene nombres de columnas categóricas, incluyendo booleanas."""
  return df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()


def get_categorical_target_associations(
  df: pd.DataFrame,
  target: str,
  categorical_columns: Optional[Sequence[str]] = None
) -> pd.Series:
  """Calcula Cramér's V entre columnas categóricas y un objetivo categórico.

  Returns:
    Serie ordenada por asociación descendente.
  """
  if target not in df.columns:
    raise KeyError(f'La columna objetivo no existe: {target}')
  columns = list(categorical_columns) if categorical_columns is not None else get_categorical_columns(df)
  columns = [column for column in columns if column != target]
  return pd.Series(
    {column: cramers_v(df[column], df[target]) for column in columns},
    name='cramers_v'
  ).sort_values(ascending=False)


def get_cramers_v_matrix(
  df: pd.DataFrame,
  categorical_columns: Optional[Sequence[str]] = None,
  max_vars: Optional[int] = 20
) -> pd.DataFrame:
  """Calcula y devuelve la matriz de asociaciones de Cramér's V."""
  columns = list(categorical_columns) if categorical_columns is not None else get_categorical_columns(df)
  if max_vars is not None and len(columns) > max_vars:
    columns = df[columns].nunique(dropna=True).sort_values(ascending=False).head(max_vars).index.tolist()

  matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
  for column_1 in columns:
    for column_2 in columns:
      matrix.loc[column_1, column_2] = cramers_v(df[column_1], df[column_2])
  return matrix


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
  """Obtiene nombres de columnas numéricas."""
  return df.select_dtypes(include=['number']).columns.tolist()


def get_numeric_correlation_matrix(
  df: pd.DataFrame,
  numeric_columns: Optional[Sequence[str]] = None,
  method: str = 'pearson',
  min_periods: int = 1
) -> pd.DataFrame:
  """Calcula una matriz numérica usando Pearson, Spearman o Kendall."""
  valid_methods = {'pearson', 'spearman', 'kendall'}
  if method not in valid_methods:
    raise ValueError(f'method debe ser uno de {sorted(valid_methods)}')
  if min_periods < 1:
    raise ValueError('min_periods debe ser mayor o igual que 1')
  columns = list(numeric_columns) if numeric_columns is not None else get_numeric_columns(df)
  return df[columns].corr(method=method, min_periods=min_periods)


def get_numeric_target_correlations(
  df: pd.DataFrame,
  target: str,
  numeric_columns: Optional[Sequence[str]] = None,
  method: str = 'pearson'
) -> pd.Series:
  """Calcula correlaciones entre columnas numéricas y un objetivo numérico."""
  if target not in df.columns:
    raise KeyError(f'La columna objetivo no existe: {target}')
  columns = list(numeric_columns) if numeric_columns is not None else get_numeric_columns(df)
  columns = [column for column in columns if column != target]
  matrix = get_numeric_correlation_matrix(df, columns + [target], method=method)
  return matrix[target].drop(labels=target).sort_values(
    key=lambda values: values.abs(), ascending=False
  )


def get_top_numeric_correlations(
  correlation_matrix: pd.DataFrame,
  threshold: float = 0.0,
  top_n: Optional[int] = None,
  absolute: bool = True
) -> pd.DataFrame:
  """Obtiene pares únicos de correlaciones que superan un umbral.

  Returns:
    DataFrame con variable_1, variable_2, correlation y absolute_correlation.
  """
  if not 0 <= threshold <= 1:
    raise ValueError('threshold debe estar entre 0 y 1')
  if not correlation_matrix.index.equals(correlation_matrix.columns):
    raise ValueError('correlation_matrix debe tener el mismo índice y columnas')
  if top_n is not None and top_n < 1:
    raise ValueError('top_n debe ser mayor o igual que 1')

  pairs: list[dict[str, object]] = []
  for index, variable_1 in enumerate(correlation_matrix.index):
    for variable_2 in correlation_matrix.columns[index + 1:]:
      correlation = correlation_matrix.loc[variable_1, variable_2]
      if pd.notna(correlation) and abs(float(correlation)) >= threshold:
        pairs.append({
          'variable_1': variable_1,
          'variable_2': variable_2,
          'correlation': float(correlation),
          'absolute_correlation': abs(float(correlation))
        })

  result = pd.DataFrame(pairs, columns=[
    'variable_1', 'variable_2', 'correlation', 'absolute_correlation'
  ])
  if not result.empty:
    result = result.sort_values(
      'absolute_correlation' if absolute else 'correlation', ascending=False
    )
  return result.head(top_n).reset_index(drop=True) if top_n is not None else result.reset_index(drop=True)


def get_top_categorical_correlations(
  association_matrix: pd.DataFrame,
  threshold: float = 0.0,
  top_n: Optional[int] = None
) -> pd.DataFrame:
  """Obtiene pares únicos de asociaciones categóricas que superan un umbral."""
  if not 0 <= threshold <= 1:
    raise ValueError('threshold debe estar entre 0 y 1')
  if not association_matrix.index.equals(association_matrix.columns):
    raise ValueError('association_matrix debe tener el mismo índice y columnas')
  if top_n is not None and top_n < 1:
    raise ValueError('top_n debe ser mayor o igual que 1')

  pairs: list[dict[str, object]] = []
  for index, variable_1 in enumerate(association_matrix.index):
    for variable_2 in association_matrix.columns[index + 1:]:
      association = association_matrix.loc[variable_1, variable_2]
      if pd.notna(association) and float(association) >= threshold:
        pairs.append({
          'variable_1': variable_1,
          'variable_2': variable_2,
          'association': float(association)
        })

  result = pd.DataFrame(pairs, columns=[
    'variable_1', 'variable_2', 'association'
  ])
  if not result.empty:
    result = result.sort_values('association', ascending=False)
  return result.head(top_n).reset_index(drop=True) if top_n is not None else result.reset_index(drop=True)


get_top_categorical_associations = get_top_categorical_correlations


def plot_categorical_countplots(
  df: pd.DataFrame,
  categorical_columns: Sequence[str],
  fig_per_row: int = 4,
  top_n: int = 15,
  title: Optional[str] = 'Distribución de variables categóricas'
) -> None:
  """Grafica countplots para columnas categóricas."""
  if fig_per_row < 1 or top_n < 1:
    raise ValueError('fig_per_row y top_n deben ser mayores o iguales que 1')
  n_cols = fig_per_row
  n_rows = max(1, math.ceil(len(categorical_columns) / n_cols))
  size = (6.5 * n_cols, 4 * n_rows) if fig_per_row > 1 else (10, 8)
  figure, axes = plt.subplots(n_rows, n_cols, figsize=size)
  axes = np.atleast_1d(axes).flatten()
  for ax, column in zip(axes, categorical_columns):
    top_categories = df[column].value_counts().nlargest(top_n).index
    label = column if df[column].nunique(dropna=True) <= top_n else f'{column} - ({top_n} de {df[column].nunique(dropna=True)})'
    sns.countplot(x=df[column], order=top_categories, ax=ax)
    ax.set_xlabel(label)
    ax.set_ylabel('Frecuencia')
    plt.setp(ax.get_xticklabels(), rotation=90)
  for ax in axes[len(categorical_columns):]:
    ax.set_visible(False)
  if title is not None:
    figure.suptitle(title)
  plt.tight_layout()
  plt.show()


def plot_correlation_heatmap(
  matrix: pd.DataFrame,
  figsize: tuple[float, float] = (12, 10),
  annot: Optional[bool] = None,
  cmap: str = 'coolwarm',
  vmin: float = -1,
  vmax: float = 1,
  cbar: bool = True,
  square: bool = False,
  mask_upper_triangle: bool = False,
  fmt: str = '.2f',
  xtick_rotation: int = 90,
  ytick_rotation: int = 0,
  title: Optional[str] = 'Matriz de correlación'
) -> None:
  """Grafica una matriz de correlación numérica configurable."""
  if annot is None:
    annot = len(matrix.columns) <= 15
  mask = np.triu(np.ones_like(matrix, dtype=bool), k=1) if mask_upper_triangle else None
  plt.figure(figsize=figsize)
  sns.heatmap(matrix, annot=annot, fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax,
              cbar=cbar, square=square, mask=mask)
  plt.xticks(rotation=xtick_rotation)
  plt.yticks(rotation=ytick_rotation)
  if title is not None:
    plt.title(title)
  plt.tight_layout()
  plt.show()


def plot_cramers_v_heatmap(
  matrix: pd.DataFrame,
  figsize: tuple[float, float] = (14, 12),
  annot: Optional[bool] = None,
  cmap: str = 'coolwarm',
  vmin: float = 0,
  vmax: float = 1,
  cbar: bool = True,
  square: bool = False,
  lower_triangle_only: bool = True,
  fmt: str = '.2f',
  xtick_rotation: int = 90,
  ytick_rotation: int = 0,
  title: Optional[str] = "Matriz de asociación - Cramér's V"
) -> None:
  """Grafica una matriz de Cramér's V previamente calculada."""
  if annot is None:
    annot = len(matrix.columns) <= 15
  mask = np.triu(np.ones_like(matrix, dtype=bool), k=1) if lower_triangle_only else None
  plt.figure(figsize=figsize)
  sns.heatmap(matrix, annot=annot, fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax,
              cbar=cbar, square=square, mask=mask)
  plt.xticks(rotation=xtick_rotation)
  plt.yticks(rotation=ytick_rotation)
  if title is not None:
    plt.title(title)
  plt.tight_layout()
  plt.show()


def plot_numeric_boxplots(
  df: pd.DataFrame,
  numeric_columns: Sequence[str],
  fig_per_row: int = 4,
  title: Optional[str] = 'Diagramas de caja'
) -> None:
  """Grafica diagramas de caja para columnas numéricas."""
  if fig_per_row < 1:
    raise ValueError('fig_per_row debe ser mayor o igual que 1')
  n_cols = fig_per_row
  n_rows = max(1, math.ceil(len(numeric_columns) / n_cols))
  size = (5 * n_cols, 4 * n_rows) if fig_per_row > 1 else (10, 8)
  figure, axes = plt.subplots(n_rows, n_cols, figsize=size)
  axes = np.atleast_1d(axes).flatten()
  for ax, column in zip(axes, numeric_columns):
    sns.boxplot(x=df[column], ax=ax)
    ax.set_xlabel(column)
  for ax in axes[len(numeric_columns):]:
    ax.set_visible(False)
  if title is not None:
    figure.suptitle(title)
  plt.tight_layout()
  plt.show()


def plot_numeric_boxplots_by_target(
  df: pd.DataFrame,
  target: str,
  numeric_columns: Sequence[str],
  fig_per_row: int = 4,
  title: Optional[str] = 'Diagramas de caja por objetivo'
) -> None:
  """Grafica boxplots numéricos separados por un objetivo categórico."""
  if fig_per_row < 1:
    raise ValueError('fig_per_row debe ser mayor o igual que 1')
  missing_columns = [
    column for column in [target, *numeric_columns] if column not in df.columns
  ]
  if missing_columns:
    raise KeyError(f'Las columnas no existen en df: {missing_columns}')
  n_cols = fig_per_row
  n_rows = max(1, math.ceil(len(numeric_columns) / n_cols))
  size = (5 * n_cols, 4 * n_rows) if fig_per_row > 1 else (10, 8)
  figure, axes = plt.subplots(n_rows, n_cols, figsize=size)
  axes = np.atleast_1d(axes).flatten()
  for ax, column in zip(axes, numeric_columns):
    sns.boxplot(data=df, x=target, y=column, ax=ax)
    ax.set_xlabel(target)
    ax.set_ylabel(column)
  for ax in axes[len(numeric_columns):]:
    ax.set_visible(False)
  if title is not None:
    figure.suptitle(title)
  plt.tight_layout()
  plt.show()


def plot_numeric_histograms(
  df: pd.DataFrame,
  numeric_columns: Sequence[str],
  fig_per_row: int = 4,
  bins: int = 30,
  kde: bool = True,
  title: Optional[str] = 'Distribución de variables numéricas'
) -> None:
  """Grafica histogramas para columnas numéricas."""
  if fig_per_row < 1 or bins < 1:
    raise ValueError('fig_per_row y bins deben ser mayores o iguales que 1')
  n_cols = fig_per_row
  n_rows = max(1, math.ceil(len(numeric_columns) / n_cols))
  size = (5 * n_cols, 4 * n_rows) if fig_per_row > 1 else (10, 8)
  figure, axes = plt.subplots(n_rows, n_cols, figsize=size)
  axes = np.atleast_1d(axes).flatten()
  for ax, column in zip(axes, numeric_columns):
    sns.histplot(df[column], bins=bins, kde=kde, ax=ax)
    ax.set_xlabel(column)
    ax.set_ylabel('Frecuencia')
  for ax in axes[len(numeric_columns):]:
    ax.set_visible(False)
  if title is not None:
    figure.suptitle(title)
  plt.tight_layout()
  plt.show()