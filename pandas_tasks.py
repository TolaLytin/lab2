"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic."""
    
    # 1. Загружаем CSV в DataFrame
    df = pd.read_csv(data.csv_path)
    
    # 2. Посчитать количество пропусков в каждом столбце
    missing_by_column = df.isnull().sum().to_dict()
    
    # 3. Число пассажиров старше 30 лет
    adults_over_30_count = int((df['Age'] > 30).sum())
    
    # 4. Средний возраст для каждого значения Pclass
    # Пропуски не считаются (по умолчанию skipna=True), ключи - int
    mean_age_by_pclass = df.groupby('Pclass')['Age'].mean().to_dict()
    mean_age_by_pclass = {int(k): float(v) for k, v in mean_age_by_pclass.items()}
    
    # 5. Доля выживших для каждого Pclass
    survival_rate_by_pclass = df.groupby('Pclass')['Survived'].mean().to_dict()
    survival_rate_by_pclass = {int(k): float(v) for k, v in survival_rate_by_pclass.items()}
    
    # 6. Пять наибольших значений Fare в порядке убывания
    highest_fares = df['Fare'].nlargest(5).tolist()
    
    return TitanicSummary(
        row_count=len(df),
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares
    )
