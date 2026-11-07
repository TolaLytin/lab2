"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt."""
    
    # Загрузка таблицы с разделителем tab и обработкой NA 
    df = pd.read_csv(data.csv_path, sep='\t', na_values=['NA'])
    
    # Признаки для корреляции
    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']
    
    # Разделение на мужчин и женщин
    men_df = df[df['Gender'] == 'Male']
    women_df = df[df['Gender'] == 'Female']
    
    # Корреляции для женщин
    women_corr = women_df[features + ['MRI_Count']].corr(method='pearson')['MRI_Count']
    women_mri_correlation = {f: women_corr[f] for f in features}
    
    # Корреляции для мужчин
    men_corr = men_df[features + ['MRI_Count']].corr(method='pearson')['MRI_Count']
    men_mri_correlation = {f: men_corr[f] for f in features}
    
    # Поиск признака с максимальным модулем корреляции среди обеих групп
    strongest_feature = None
    max_corr = -1.0
    
    for f in features:
        w_val = abs(women_mri_correlation[f]) if not pd.isna(women_mri_correlation[f]) else 0
        m_val = abs(men_mri_correlation[f]) if not pd.isna(men_mri_correlation[f]) else 0
        
        current_max = max(w_val, m_val)
        if current_max > max_corr:
            max_corr = current_max
            strongest_feature = f
            
    return BrainCorrelationSummary(
        men_count=len(men_df),
        women_count=len(women_df),
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_feature
    )
