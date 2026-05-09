"""Hotel review dataset loader for Chapter 4.

The dataset `traversaal-ai-hackathon/hotel_datasets` lives on the HuggingFace Hub
and contains hotel reviews from many cities. The book chapter focuses on Paris.
"""

from __future__ import annotations

import pandas as pd
from datasets import load_dataset

DATASET_NAME = "traversaal-ai-hackathon/hotel_datasets"


def load_hotel_reviews() -> pd.DataFrame:
    dataset = load_dataset(DATASET_NAME)
    return pd.DataFrame(dataset["train"])


def filter_city(df: pd.DataFrame, city: str = "Paris") -> pd.DataFrame:
    return df.loc[df.locality == city].drop_duplicates().reset_index(drop=True)


def load_paris_reviews() -> pd.DataFrame:
    return filter_city(load_hotel_reviews(), city="Paris")
