import os
import numpy as np
import pandas as pd
from datetime import datetime
from interface import Metric


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


def cmap(value, threshold=1):
    if value < threshold:
        color = 'lightgreen'
    else:
        color = 'red'
    return f'background-color : {color}'


def create_report(course_id=None, metrics_list=None, data_tables=None):
    course_module = data_tables['course_module']
    course_element = data_tables['course_element']
    output = pd.DataFrame(columns=['element_id'])
    styled_tabels = []
    thresholds = []
    n_metrics = len(metrics_list)
    for metric in metrics_list:
        thresholds.append(metric.threshold)
        output = pd.merge(output, metric.evaluate(course_id=course_id), on='element_id', how='outer')
    modules_list = list(course_module[course_module.course_id == course_id].id)
    if not os.path.isdir(str(course_id)):
        os.mkdir(str(course_id))
    for module in modules_list:
        current_module_tasks = course_element[course_element.module_id == module].sort_values(by='position')['element_id']
        styled = pd.merge(current_module_tasks, output, on='element_id')
        for i in range(1, n_metrics + 1):
            if i == 1:
                styled = styled.style.map(cmap, subset=[styled.columns[i]], **{'threshold' : thresholds[i - 1]})
            else:
                styled = styled.map(cmap, subset=[styled.columns[i]], **{'threshold' : thresholds[i - 1]})
        styled_tabels.append(styled)
        styled.to_excel(os.path.join(str(course_id), f'{module}.xlsx'), index=False)
    return styled_tabels
