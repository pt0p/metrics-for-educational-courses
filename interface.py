import pandas as pd
from pandas import DataFrame
from typing import List, Dict, Union, Tuple
from abc import ABC, abstractmethod


class Metric(ABC):
    def __init__(self, metric_name: str, threshold,
                 parameters: Dict[str, float],
                 data_tables: Union[List[DataFrame], Dict[str, DataFrame]]) -> None:
        self.metric_name = metric_name
        self.threshold = threshold
        self.parameters = parameters
        self.data_tables = data_tables


    @abstractmethod
    def evaluate(self, *args, **kwargs):# -> DataFrame({'id': int, self.metric_name: float}):
        pass


class MetricsCalculator:
    def __init__(self, metrics_list: List[Metric], metrics_args: List[Tuple[List, Dict]]):
        self.metrics_list = metrics_list
        self.metrics_args = metrics_args


    def calculate_metrics(self):
        args, kwargs = self.metrics_args[0]
        results = self.metrics_list[0].evaluate(*args, **kwargs)
        for metric, metric_args in zip(self.metrics_list[1:], self.metrics_args[1:]):
            args, kwargs = metric_args
            results = pd.merge(results, metric.evaluate(*args, **kwargs), on='id', how='outer')
        return results
