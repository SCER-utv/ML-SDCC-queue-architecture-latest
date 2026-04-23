from dataclasses import dataclass
from typing import Optional


@dataclass
class JobPaths:
    # Definitive paths
    # Workers and inferencer will always read from here
    train_url: str
    test_url: str
    metrics_key: str

    # This field is filled only if train/test files do not exist already
    # and the master neeeds this raw file to generate them
    raw_source_to_split: Optional[str] = None