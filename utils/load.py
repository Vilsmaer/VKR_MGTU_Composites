"""
Модуль загрузки и первичной обработки данных.
Поддерживает CSV, XLSX, объединение файлов и редактирование колонок.
"""
import pandas as pd
import os
from typing import List, Optional, Tuple, Literal


def load_file(file_path: str) -> pd.DataFrame:
    """
    Загружает файл CSV или XLSX в DataFrame.
    Корректно обрабатывает кириллицу (UTF-8, CP1251).
    
    Args:
        file_path: Путь к файлу.
        
    Returns:
        DataFrame с данными файла.
        
    Raises:
        ValueError: Если формат файла не поддерживается.
        FileNotFoundError: Если файл не найден.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.csv':
            # Пробуем несколько распространенных кодировок для кириллицы
            encodings_to_try = ['utf-8', 'utf-8-sig', 'cp1251', 'latin1']
            df = None
            
            for encoding in encodings_to_try:
                try:
                    # Сначала пробуем с авто-определением разделителя (sep=None включает engine='python')
                    df = pd.read_csv(file_path, encoding=encoding, sep=None, engine='python')
                    break  # Если успешно, выходим из цикла
                except UnicodeDecodeError:
                    continue  # Пробуем следующую кодировку
                except Exception:
                    # Если sep=None не сработал, пробуем стандартные разделители
                    for sep in [',', ';', '\t']:
                        try:
                            df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                            break
                        except Exception:
                            continue
                    if df is not None:
                        break
            
            if df is None:
                # Если все кодировки не подошли, пробуем с игнорированием ошибок
                df = pd.read_csv(file_path, encoding='utf-8', errors='replace', sep=None, engine='python')
                
        elif ext in ['.xlsx', '.xls']:
            # openpyxl корректно работает с кириллицей по умолчанию
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {ext}. Используйте .csv или .xlsx")
        
        # Базовая валидация типов
        df = _validate_column_types(df)
        return df
        
    except Exception as e:
        raise RuntimeError(f"Ошибка при загрузке файла {file_path}: {str(e)}")


def _validate_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Простая валидация и приведение типов колонок.
    Удаляет BOM (Byte Order Mark) из имен колонок для корректной работы с UTF-8-SIG.
    # TODO: Добавить более сложную логику определения типов (даты, категории)
    """
    # Удаляем BOM из имен колонок если он есть
    new_columns = []
    for col in df.columns:
        if col.startswith('\ufeff'):
            new_columns.append(col[1:])  # Удаляем первый символ BOM
        else:
            new_columns.append(col)
    
    if new_columns != list(df.columns):
        df.columns = new_columns
    
    for col in df.columns:
        # Попытка привести к числовому типу, если возможно
        if df[col].dtype == 'object':
            try:
                # Пробуем конвертировать в numeric, но не меняем колонку если есть строки
                pd.to_numeric(df[col], errors='raise')
            except (ValueError, TypeError):
                pass  # Оставляем как объект (строка/категория)
    return df


def preview_random_rows(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Возвращает n случайных строк из DataFrame для превью.
    
    Args:
        df: Исходный DataFrame.
        n: Количество строк для показа.
        
    Returns:
        DataFrame с n случайными строками.
    """
    if len(df) == 0:
        return df
    return df.sample(n=min(n, len(df)), random_state=42)


def concatenate_dataframes(
    dfs: List[pd.DataFrame], 
    axis: Literal[0, 1] = 0,
    ignore_index: bool = True
) -> pd.DataFrame:
    """
    Объединяет список DataFrame по оси 0 (строки) или 1 (колонки).
    
    Args:
        dfs: Список DataFrame для объединения.
        axis: 0 - вертикальное объединение, 1 - горизонтальное.
        ignore_index: Сбросить индексы при axis=0.
        
    Returns:
        Объединенный DataFrame.
        
    Raises:
        ValueError: Если список пуст или содержит не-DataFrame.
    """
    if not dfs:
        raise ValueError("Список DataFrame пуст")
    
    if any(not isinstance(df, pd.DataFrame) for df in dfs):
        raise ValueError("Все элементы должны быть DataFrame")
    
    if len(dfs) == 1:
        return dfs[0].copy()
    
    return pd.concat(dfs, axis=axis, ignore_index=ignore_index)


def merge_dataframes(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
    on: Optional[str] = None,
    how: Literal['inner', 'left', 'right', 'outer'] = 'inner'
) -> pd.DataFrame:
    """
    Объединяет два DataFrame по ключевой колонке.
    
    Args:
        df_left: Левый DataFrame.
        df_right: Правый DataFrame.
        left_on: Колонка в левом DF.
        right_on: Колонка в правом DF.
        on: Общая колонка (если имена совпадают).
        how: Тип соединения.
        
    Returns:
        Объединенный DataFrame.
    """
    # Обработка дубликатов имен колонок
    common_cols = set(df_left.columns) & set(df_right.columns)
    if common_cols and not on:
        # Удаляем ключевую колонку из правого DF если она будет использоваться
        key_col = left_on or right_on
        if key_col and key_col in common_cols:
            common_cols.discard(key_col)
        
        # Добавляем суффиксы к дублирующимся колонкам
        suffixes = ('_left', '_right')
        return pd.merge(
            df_left, df_right,
            left_on=left_on, right_on=right_on, on=on,
            how=how, suffixes=suffixes
        )
    
    return pd.merge(
        df_left, df_right,
        left_on=left_on, right_on=right_on, on=on,
        how=how
    )


def merge_by_inner_index(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    how: Literal['inner', 'left', 'right', 'outer'] = 'inner'
) -> pd.DataFrame:
    """
    Объединяет DataFrame по внутренним индексам.
    # TODO: Добавить опцию сброса индексов после merge
    
    Args:
        df_left: Левый DataFrame.
        df_right: Правый DataFrame.
        how: Тип соединения.
        
    Returns:
        Объединенный DataFrame.
    """
    return pd.merge(
        df_left, df_right,
        left_index=True, right_index=True,
        how=how
    )


def rename_columns(df: pd.DataFrame, new_names: dict) -> pd.DataFrame:
    """
    Переименовывает колонки в DataFrame.
    
    Args:
        df: Исходный DataFrame.
        new_names: Словарь {старое_имя: новое_имя}.
        
    Returns:
        DataFrame с переименованными колонками.
        
    Raises:
        ValueError: Если новые имена содержат дубликаты.
    """
    # Проверка на дубликаты новых имен
    unique_new_names = set(new_names.values())
    if len(unique_new_names) != len(new_names):
        raise ValueError("Новые имена колонок содержат дубликаты")
    
    # Проверка на конфликт с существующими колонками (которые не переименовываются)
    existing_cols = set(df.columns) - set(new_names.keys())
    conflicting = existing_cols & unique_new_names
    if conflicting:
        raise ValueError(f"Новые имена конфликтуют с существующими колонками: {conflicting}")
    
    return df.rename(columns=new_names)


def get_duplicate_columns(df: pd.DataFrame) -> List[str]:
    """
    Находит колонки с дублирующимися именами.
    
    Args:
        df: DataFrame для проверки.
        
    Returns:
        Список имен колонок, которые дублируются.
    """
    seen = set()
    duplicates = set()
    for col in df.columns:
        if col in seen:
            duplicates.add(col)
        else:
            seen.add(col)
    return list(duplicates)


def export_to_csv(df: pd.DataFrame, output_path: str, index: bool = False) -> str:
    """
    Экспортирует DataFrame в CSV файл с корректной кодировкой UTF-8-SIG.
    UTF-8-SIG (BOM) обеспечивает правильное отображение кириллицы в Excel.
    
    Args:
        df: DataFrame для экспорта.
        output_path: Путь для сохранения.
        index: Сохранять ли индекс.
        
    Returns:
        Путь к сохраненному файлу.
    """
    try:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        # encoding='utf-8-sig' добавляет BOM для корректного отображения в Excel
        df.to_csv(output_path, index=index, encoding='utf-8-sig', sep=';')
        return output_path
    except Exception as e:
        raise RuntimeError(f"Ошибка при экспорте в CSV: {str(e)}")
