import pandas as pd
from pandas import DataFrame
from typing import List, Dict, Union, Tuple
from abc import ABC, abstractmethod
from datetime import datetime

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

class N_tries(Metric):
    def evaluate(self, course_id=None):
        tasks = self.data_tables['user_element_progress'][(self.data_tables['user_element_progress']["course_element_type"] == "task") &
                                                       (self.data_tables['user_element_progress']["achieve_reason"] != "transferred") &
                                                        (self.data_tables['user_element_progress']["course_id"] == course_id)]
        elements = np.unique(np.array(tasks['course_element_id']))
        N = self.parameters['N']
        dict_metric = {}
        for task in elements:
            count = 0
            task_tries = tasks[(tasks['course_element_id'] == task) & (tasks['tries_count'] >= N)]
            task_users = np.unique(np.array(task_tries['user_id']))
            for user in task_users:
                user_tries = task_tries[task_tries['user_id'] == user]
                tries = np.array(user_tries['tries_count'])
                count += tries.size != 0
            unique_tasks = np.unique(np.array(tasks[tasks['course_element_id'] == task]['user_id']))
            dict_metric[task] = int(100 * count / unique_tasks.size) / 100
        return pd.DataFrame({'element_id' : dict_metric.keys(), self.metric_name : dict_metric.values()})

class diff_tries(Metric):
    def evaluate(self, course_id=None):
        tasks = self.data_tables['user_element_progress'][(self.data_tables['user_element_progress']["course_element_type"] == "task") &
                                                       (self.data_tables['user_element_progress']["achieve_reason"] != "transferred") &
                                                        (self.data_tables['user_element_progress']["course_id"] == course_id)]
        tasks = tasks[tasks['tries_count'] > 0]
        elements = np.unique(np.array(tasks['course_element_id']))
        dict_metric = {}
        inf = 1000
        for task in elements:
            dict_metric[task] = []
            task_tries = tasks[tasks['course_element_id'] == task]
            task_users = np.unique(np.array(task_tries['user_id']))
            for user in task_users:
                user_tries = tasks[tasks['user_id'] == user]
                modules = user_tries[user_tries['course_element_id'] == task]['course_module_id'].to_string(index=False).split('\n')
                metric_for_user = -inf
                for module in modules:
                    user_tries_per_module = user_tries[user_tries['course_module_id'] == int(module)]
                    tries = np.array(user_tries_per_module['tries_count'])
                    metric_for_user = max(int(user_tries[(user_tries['course_element_id'] == task) & (user_tries['course_module_id'] == int(module))]['tries_count'].to_string(index=False)) - tries.mean(), metric_for_user)
                dict_metric[task].append(metric_for_user)
            dict_metric[task] = round(np.mean(dict_metric[task]), 2)
        return pd.DataFrame({'element_id' : dict_metric.keys(), self.metric_name : dict_metric.values()})

class percentage_tries(Metric):
    def evaluate(self, course_id=None):
        tasks = self.data_tables['user_element_progress'][(self.data_tables['user_element_progress']["course_element_type"] == "task") &
                                                       (self.data_tables['user_element_progress']["achieve_reason"] != "transferred") &
                                                          (self.data_tables['user_element_progress']["course_id"] == course_id)]
        tasks = tasks[tasks['tries_count'] > 0]
        elements = np.unique(np.array(tasks['course_element_id']))
        dict_metric = {}
        inf = 1000
        for task in elements:
            dict_metric[task] = []
            task_tries = tasks[tasks['course_element_id'] == task]
            task_users = np.unique(np.array(task_tries['user_id']))
            for user in task_users:
                user_tries = tasks[tasks['user_id'] == user]
                modules = user_tries[user_tries['course_element_id'] == task]['course_module_id'].to_string(index=False).split('\n')
                metric_for_user = -inf
                for module in modules:
                    user_tries_per_module = user_tries[user_tries['course_module_id'] == int(module)]
                    tries = np.array(user_tries_per_module['tries_count'])
                    metric_for_user = max(int(user_tries[(user_tries['course_element_id'] == task) & (user_tries['course_module_id'] == int(module))]['tries_count'].to_string(index=False))/tries.sum(), metric_for_user)
                dict_metric[task].append(metric_for_user)
            dict_metric[task] = round(np.mean(dict_metric[task]), 2)
        return pd.DataFrame({'element_id' : dict_metric.keys(), self.metric_name : dict_metric.values()})


## EXAMPLES
#N_tries_object = N_tries(metric_name='N_tries', threshold=10, parameters={'N' : 7}, data_tables=ml_course_info)
#diff_tries_object = diff_tries(metric_name='diff_tries', threshold=1, parameters={}, data_tables=ml_course_info)
#percentage_tries_object = percentage_tries(metric_name='percentage_tries', threshold=0.9, parameters={}, data_tables=ml_course_info)
