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
            dict_metric[task] = int(100 * count / unique_tasks.size)
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

class TaskTime(Metric):
    def evaluate(self, course_id=None):
        metric = self.parameters.get('metric', 'median')
        if metric == 'median':
            metric = np.median
        else:
            metric = np.mean
        user_session = self.parameters.get('user_session', 40)
        course_module = self.data_tables['course_module']
        course_element = self.data_tables['course_element']
        user_element_progress = self.data_tables['user_element_progress']
        solution_log = self.data_tables['solution_log'].copy()
        solution_log.submission_time = solution_log.submission_time.apply(lambda x: datetime.fromisoformat(x))
        # Выбираю те записи, которые соответствуют данному курсу
        modules_list = list(course_module[course_module.course_id == course_id].id)
        user_element_progress = user_element_progress[user_element_progress.course_module_id.isin(modules_list)]
        course_element = course_element[course_element.module_id.isin(modules_list)]
        # Выбираю решенные + задачи в данном модуле
        solved_tasks = user_element_progress[user_element_progress.course_element_type == 'task']
        solved_tasks = solved_tasks[solved_tasks.achieve_reason == 'solved']
        # список задач в данном модуле:
        tasks_list = np.unique(course_element[course_element.element_type == 'task'].element_id)
        output = []
        for task in tasks_list:
            users_time = []
            progress_ids = solved_tasks[solved_tasks.course_element_id == task].id
            for id in progress_ids:
                df = solution_log[solution_log.element_progress_id == id].sort_values('submission_time',
                                                                                               ascending=False).submission_time

                user_time = 0
                for i in range(len(df) - 1):
                    stop = df.iloc[i]
                    start = df.iloc[i + 1]
                    attempt_time = (stop - start).total_seconds() / 60
                    if 0 < attempt_time < user_session:
                        user_time += attempt_time
                users_time.append(user_time)
            output.append(metric(users_time))
        ids = course_element.set_index('element_id').loc[tasks_list, 'id']
        return pd.DataFrame({'element_id' : tasks_list, self.metric_name : output})


class TaskTimeDeviation(Metric):
    def evaluate(self, course_id=None):
        metric = self.parameters.get('metric', 'median')
        if metric == 'mean':
            task_metric = np.mean
        else:
            task_metric = np.median
        user_session = self.parameters.get('user_session', 40)
        course_module = self.data_tables['course_module']
        course_element = self.data_tables['course_element']
        user_element_progress = self.data_tables['user_element_progress']
        solution_log = self.data_tables['solution_log'].copy()
        solution_log.submission_time = solution_log.submission_time.apply(lambda x: datetime.fromisoformat(x))
        # Выбираю те записи, которые соответствуют данному модулю
        modules_list = list(course_module[course_module.course_id == course_id].id)
        user_element_progress = user_element_progress[user_element_progress.course_module_id.isin(modules_list)]
        course_element = course_element[course_element.module_id.isin(modules_list)]
        # Выбираю решенные + задачи в данном модуле
        solved_tasks = user_element_progress[user_element_progress.course_element_type == 'task']
        solved_tasks = solved_tasks[solved_tasks.achieve_reason == 'solved']
        # список задач в данном модуле:
        tasks_list = np.unique(course_element[course_element.element_type == 'task'].element_id)
        output = []
        users_time = []
        for id in solved_tasks.id:
            df = solution_log[solution_log.element_progress_id == id].sort_values('tries_count',
                                                                                 ascending=False).submission_time
            user_time = 0
            for i in range(len(df) - 1):
                stop = df.iloc[i]
                start = df.iloc[i + 1]
                attempt_time = (stop - start).total_seconds() / 60
                if 0 < attempt_time < user_session:
                    user_time += attempt_time
            users_time.append(user_time)
        users_time = pd.DataFrame({'id' : list(solved_tasks.id), 'user_time' : users_time})
        solved_tasks = pd.merge(solved_tasks, users_time, on='id')
        median_module_time = []
        module_progress_ids = np.unique(solved_tasks.module_progress_id)
        for id in module_progress_ids:
            median_module_time.append(task_metric(solved_tasks[solved_tasks.module_progress_id == id].user_time))
        median_module_time = pd.DataFrame({'module_progress_id' : module_progress_ids, 'median_module_time' : median_module_time})
        solved_tasks = pd.merge(solved_tasks, median_module_time, on='module_progress_id')
        solved_tasks.user_time = solved_tasks.user_time - solved_tasks.median_module_time
        if metric == 'median':
            output = solved_tasks.groupby('course_element_id')['user_time'].median()
        else:
            output = solved_tasks.groupby('course_element_id')['user_time'].mean()
        return pd.DataFrame({'element_id' : output.index, self.metric_name : list(output)})
