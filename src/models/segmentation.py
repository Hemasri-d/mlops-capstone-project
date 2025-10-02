import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.cluster import KMeans


def prepare_segmentation_features(customer_metrics: pd.DataFrame) -> pd.DataFrame:
    features = customer_metrics[[
        'total_spent',
        'avg_transaction_value',
        'transaction_count',
        'days_since_last_purchase',
        'categories_purchased',
        'malls_visited',
    ]].fillna(0.0)

    X = features.copy()
    X['total_spent'] = np.log1p(X['total_spent'])
    X['avg_transaction_value'] = np.log1p(X['avg_transaction_value'])
    X['transaction_count'] = np.log1p(X['transaction_count'])
    X['days_since_last_purchase'] = np.log1p(X['days_since_last_purchase'].clip(lower=0))
    return X


def train_kmeans_segmentation(customer_metrics: pd.DataFrame, n_clusters: int = 5, random_state: int = 42) -> Tuple[np.ndarray, float]:
    X = prepare_segmentation_features(customer_metrics)
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = model.fit_predict(X.values)
    return labels, float(model.inertia_)


