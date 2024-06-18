import numpy as np
import pandas as pd
from datetime import datetime
from interface import Metric
from datetime import timedelta

# Element metrics:
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


class MeanTriesCount(Metric):
    def filter_df(self, df: pd.DataFrame, outlier_threshhold: float) -> pd.DataFrame:
        """Leave only tasks without transferred progress; remove outliers"""
        df = df[df["course_element_type"] == "task"]
        df = df[df["achieve_reason"] != "transferred"]
        df_filtered = df[df["tries_count"] < outlier_threshhold]
        df_filtered = df_filtered[df_filtered["tries_count"] > 0]
        return df_filtered

    def evaluate(self, course_id=None, *args, **kwargs):
        user_element_progress = self.data_tables["user_element_progress"]
        if course_id != None:
            user_element_progress = user_element_progress[
                self.data_tables["user_element_progress"]["course_id"] == course_id
            ]
        outlier_threshold = self.parameters["outlier"]
        df_filtered = self.filter_df(user_element_progress, outlier_threshold)
        mean_tries = (
            df_filtered[["course_element_id", "tries_count"]]
            .groupby(by="course_element_id")
            .mean()["tries_count"]
        )
        return pd.DataFrame(
            {"element_id": mean_tries.index, self.metric_name: mean_tries.values}
        )


class TriesStd(Metric):
    def filter_df(self, df: pd.DataFrame, outlier_threshhold: float) -> pd.DataFrame:
        """Leave only tasks without transferred progress; remove outliers"""
        df = df[df["course_element_type"] == "task"]
        df = df[df["achieve_reason"] != "transferred"]
        df_filtered = df[df["tries_count"] < outlier_threshhold]
        df_filtered = df_filtered[df_filtered["tries_count"] > 0]
        return df_filtered

    def evaluate(self, course_id=None, *args, **kwargs):
        outlier_threshold = self.parameters["outlier"]
        user_element_progress = self.data_tables["user_element_progress"]
        if course_id != None:
            user_element_progress = user_element_progress[
                self.data_tables["user_element_progress"]["course_id"] == course_id
            ]
        df_filtered = self.filter_df(user_element_progress, outlier_threshold)
        tries_std = (
            df_filtered[["course_element_id", "tries_count"]]
            .groupby("course_element_id")
            .std()["tries_count"]
        )
        return pd.DataFrame(
            {"element_id": tries_std.index, self.metric_name: tries_std.values}
        )


class SkipsPercentage(Metric):
    def evaluate(self, course_id=None, *args, **kwargs):
        course_element = self.data_tables["course_element"]
        course_element = course_element.rename(
            columns={
                "module_id": "course_module_id",
                "element_id": "course_element_id",
                "element_type": "course_element_type",
            }
        )
        user_element_progress = self.data_tables["user_element_progress"]
        if course_id != None:
            user_element_progress = user_element_progress[
                self.data_tables["user_element_progress"]["course_id"] == course_id
            ]
        df = user_element_progress[
            user_element_progress["course_element_type"] == "task"
        ]
        df = df[df["achieve_reason"] != "transferred"]
        df_for_skips = pd.merge(
            df,
            course_element,
            how="inner",
            on=["course_module_id", "course_element_type", "course_element_id"],
        )

        groups_with_skipped_col = []
        columns = ["user_id", "course_element_id", "skipped"]

        for groupname, group in df_for_skips.groupby(["user_id", "course_module_id"]):
            group = self.add_skipped_column(group)[columns]
            groups_with_skipped_col.append(group)

        skips_data = pd.concat(groups_with_skipped_col)
        skips_count = skips_data.groupby(by="course_element_id").sum()["skipped"]
        # запись о пользователе-элементе в user_element_progress есть только тогда, когда задача открыта пользователю
        user_count = skips_data.groupby(by="course_element_id").count()["user_id"]

        skips_percentage = skips_count / user_count
        return pd.DataFrame(
            {
                "element_id": skips_percentage.index,
                self.metric_name: skips_percentage.values,
            }
        )

    def add_skipped_column(self, group):
        """Добавляет колонку skipped о пропуске задачи для подмножества user_element_progress с фиксированными module_id, user_id"""
        group = group.sort_values(by="position")  # для корректности на NaN
        group = group.sort_values(by="time_achieved")
        group["shift_position"] = group["position"].shift()
        group["max_solved_before"] = (
            group["shift_position"].rolling(2, min_periods=1).max()
        )
        # номер задачи меньше максимального номера уже решенной задачи => ее пропускали
        group["skipped"] = group["position"].le(group["max_solved_before"])
        # group['skipped'] = group['position'].le(group['position'].shift().rolling(2, min_periods=1).max())
        return group


class LostPercentage(Metric):
    def evaluate(self, course_id=None, *args, **kwargs):
        user_element_progress = self.data_tables["user_element_progress"]
        if course_id != None:
            user_element_progress = user_element_progress[
                self.data_tables["user_element_progress"]["course_id"] == course_id
            ]
        df = user_element_progress[
            user_element_progress["course_element_type"] == "task"
        ]
        df = df[df["achieve_reason"] != "transferred"]
        course_element = self.data_tables["course_element"]
        course_element = course_element.rename(
            columns={
                "module_id": "course_module_id",
                "element_id": "course_element_id",
                "element_type": "course_element_type",
            }
        )
        df_for_lost = pd.merge(
            user_element_progress,
            course_element,
            how="inner",
            on=["course_module_id", "course_element_type", "course_element_id"],
        )

        columns = [
            "user_id",
            "course_module_id",
            "course_element_type",
            "course_element_id",
            "position",
            "is_achieved",
        ]
        df_for_lost = df_for_lost[columns]
        groups = []

        for groupname, group in df_for_lost.groupby(["user_id", "course_module_id"]):
            group = group.sort_values("position")
            group["prev_is_task"] = "task" == group["course_element_type"].shift()
            group["prev_is_solved"] = group["is_achieved"].shift()
            # оставим только информацию о задачах
            group = group[group["course_element_type"] == "task"]
            # оставим только задачи, идущие в модуле сразу после других задач (а не видео/текста)
            group = group[group["prev_is_task"]]
            groups.append(group)

        lost_data = pd.concat(groups)
        lost_data["is_achieved"] = lost_data["is_achieved"].fillna(False)
        lost_data["is_lost"] = lost_data["prev_is_solved"] & (~lost_data["is_achieved"])
        counts_df = (
            lost_data[["course_element_id", "prev_is_solved", "is_lost"]]
            .groupby("course_element_id")
            .sum()
        )
        lost_count = counts_df["is_lost"]
        solved_prev_count = counts_df["prev_is_solved"]
        lost_percentage = lost_count / solved_prev_count

        return pd.DataFrame(
            {
                "element_id": lost_percentage.index,
                self.metric_name: lost_percentage.values,
            }
        )


class GuessedPercentage(Metric):
    def evaluate(self, course_id=None, *args, **kwargs):
        user_element_progress = self.data_tables["user_element_progress"]
        if course_id != None:
            user_element_progress = user_element_progress[
                self.data_tables["user_element_progress"]["course_id"] == course_id
            ]
        df = user_element_progress[
            user_element_progress["course_element_type"] == "task"
        ]
        df = df[df["achieve_reason"] != "transferred"]
        df_tried = df[df["tries_count"] > 0]
        solution_log = self.data_tables["solution_log"].copy()
        solution_log["submission_time"] = pd.to_datetime(
            solution_log["submission_time"]
        )

        element_progress_ids = solution_log["element_progress_id"].unique()
        # по element_progress_id записано, угадывалась ли данная задача данным пользователем
        progress_id_guessed = pd.Series(
            data=[False] * len(element_progress_ids), index=element_progress_ids
        )

        max_time_delta = pd.Timedelta(seconds=10)
        N = 5

        for group_id, group in solution_log.groupby("element_progress_id"):
            if len(group) < N:
                # если N попыток не сделано, то "угадывания" точно не было
                progress_id_guessed[group_id] = False
                continue
            group = group.sort_values("submission_time")
            # group['prev_time'] = group['submission_time'].shift()
            # group['time_delta'] = group['submission_time'] - group['submission_time'].shift()
            group["is_guess"] = (
                group["submission_time"] - group["submission_time"].shift()
            ) <= max_time_delta
            group["is_N_guesses"] = group["is_guess"].rolling(N, min_periods=1).min()
            progress_id_guessed[group_id] = group["is_N_guesses"].any()

        guess_data = pd.DataFrame(
            {"id": progress_id_guessed.index, "guessed": progress_id_guessed.values}
        )
        df_tried = pd.merge(df_tried, guess_data, how="left", on="id")
        # количество угадывавших
        guessed_count = (
            df_tried[["course_element_id", "guessed"]]
            .groupby("course_element_id")
            .sum()["guessed"]
        )
        # количество пытавшихся решить задачу
        tried_count = (
            df_tried[["course_element_id", "tries_count"]]
            .groupby("course_element_id")
            .count()["tries_count"]
        )
        guessed_percentage = guessed_count / tried_count
        return pd.DataFrame(
            {
                "element_id": guessed_percentage.index,
                self.metric_name: guessed_percentage.values,
            }
        )


class SolvedPercentage(Metric):
    def evaluate(self, course_id=None):
        user_element_progress = self.data_tables["user_element_progress"]
        if course_id != None:
            user_element_progress = user_element_progress[
                self.data_tables["user_element_progress"]["course_id"] == course_id
            ]
        df = user_element_progress[
            user_element_progress["course_element_type"] == "task"
        ]
        df = df[df["achieve_reason"] != "transferred"]
        df = df[df["tries_count"] > 0]
        df["is_achieved"] = df["is_achieved"].fillna(False).astype(int)
        grouped = (
            df[["course_element_id", "is_achieved", "user_id"]]
            .groupby("course_element_id")
            .agg({"user_id": "count", "is_achieved": "sum"})
            .reset_index()
        )
        solved_percentage = pd.DataFrame(
            {
                "element_id": grouped["course_element_id"],
                self.metric_name: grouped["is_achieved"] / grouped["user_id"],
            }
        )
        return solved_percentage


class NTries(Metric):
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
            dict_metric[task] = count / unique_tasks.size
        return pd.DataFrame({'element_id' : dict_metric.keys(), self.metric_name : dict_metric.values()})


class DiffTries(Metric):
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


class PercentageTries(Metric):
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


# Module metrics:

def filter_module_progress(
    module_progress: pd.DataFrame, element_progress: pd.DataFrame
):
    """Убирает прогрессы по модулям, в которых был autograde; по которым не было совершено попыток"""
    df = module_progress[module_progress["achieve_reason"] != "autograde"]
    element_progress["is_transferred"] = (
        element_progress["achieve_reason"] == "transferred"
    ).astype(int)
    tries_sum = (
        element_progress[
            [
                "course_id",
                "user_id",
                "course_module_id",
                "course_element_id",
                "tries_count",
                "is_transferred",
            ]
        ]
        .groupby(["course_id", "course_module_id", "user_id"])
        .sum()
        .reset_index()
    )
    df = df.merge(
        tries_sum, on=["course_id", "course_module_id", "user_id"], how="left"
    )
    df = df[df["tries_count"] > 0]
    return df


class ModuleUserCount(Metric):
    def evaluate(self, course_id=None):
        user_module_progress = self.data_tables["user_module_progress"].copy()
        user_element_progress = self.data_tables["user_element_progress"].copy()
        if course_id != None:
            user_module_progress = user_module_progress[
                user_module_progress["course_id"] == course_id
            ]
            user_element_progress = user_element_progress[
                user_element_progress["course_id"] == course_id
            ]
        df = filter_module_progress(user_module_progress, user_element_progress)
        user_count = (
            df.groupby("course_module_id")
            .agg(user_count=("user_id", pd.Series.nunique))
            .reset_index()
            .rename(
                columns={
                    "course_module_id": "module_id",
                    "user_count": self.metric_name,
                }
            )
        )
        return user_count


class ModuleAchievedPercentage(Metric):
    def evaluate(self, course_id=None):
        user_module_progress = self.data_tables["user_module_progress"].copy()
        user_element_progress = self.data_tables["user_element_progress"].copy()
        if course_id != None:
            user_module_progress = user_module_progress[
                user_module_progress["course_id"] == course_id
            ]
            user_element_progress = user_element_progress[
                user_element_progress["course_id"] == course_id
            ]
        df = filter_module_progress(user_module_progress, user_element_progress)

        agg_df = (
            df[["course_module_id", "is_achieved", "user_id"]]
            .groupby(["course_module_id"])
            .agg({"user_id": "count", "is_achieved": "sum"})
            .reset_index()
        )
        achieved_percentage = pd.DataFrame(
            {
                "module_id": agg_df["course_module_id"],
                self.metric_name: agg_df["is_achieved"] / agg_df["user_id"],
            }
        )
        return achieved_percentage


class ModuleTime(Metric):
    def evaluate(self, course_id=None):
        user_module_progress = self.data_tables["user_module_progress"].copy()
        user_element_progress = self.data_tables["user_element_progress"].copy()
        solution_log = self.data_tables["solution_log"].copy()
        if course_id != None:
            user_module_progress = user_module_progress[
                user_module_progress["course_id"] == course_id
            ]
            user_element_progress = user_element_progress[
                user_element_progress["course_id"] == course_id
            ]

        element_progress = user_element_progress[
            ["user_id", "course_element_type", "course_module_id", "time_achieved"]
        ]
        element_progress["element_progress_id"] = user_element_progress["id"]
        log = solution_log[["id", "element_progress_id", "submission_time"]]
        df = log.merge(element_progress, on="element_progress_id", how="outer")
        columns = ["course_module_id", "user_id", "action_time"]

        no_tasks = df[df["course_element_type"] != "task"]
        no_tasks["action_time"] = no_tasks["time_achieved"]
        no_tasks = no_tasks[columns].dropna()

        tasks = df[df["course_element_type"] == "task"]
        tasks["action_time"] = tasks["submission_time"]
        tasks = tasks[columns].dropna()

        actions = pd.concat([tasks, no_tasks])
        max_delta = timedelta(minutes=self.parameters["max_timedelta_min"])
        module_times = {}

        for name, group in actions.groupby(["course_module_id", "user_id"]):
            group = group.sort_values(by="action_time")
            group["action_time"] = pd.to_datetime(group["action_time"])
            group["time_delta"] = group["action_time"] - group["action_time"].shift()
            group = group[group["time_delta"] < max_delta]
            if name[0] in module_times.keys():
                module_times[name[0]].append(group["time_delta"].sum())
            else:
                module_times[name[0]] = [group["time_delta"].sum()]

        for module in module_times.keys():
            module_times[module] = np.mean(module_times[module])

        mean_module_time = pd.DataFrame(
            {"module_id": module_times.keys(), self.metric_name: module_times.values()}
        )
        mean_module_time[self.metric_name] = mean_module_time[self.metric_name].apply(
            lambda x: x.total_seconds() / 60
        )
        return mean_module_time


class ModuleMeanTries(Metric):
    def evaluate(self, course_id=None):
        user_element_progress = self.data_tables["user_element_progress"].copy()
        if course_id != None:
            user_element_progress = user_element_progress[user_element_progress.course_id == course_id]
        tasks = user_element_progress[(user_element_progress["course_element_type"] == "task") &
                                                (user_element_progress["achieve_reason"] != "transferred")]
        modules = np.unique(np.array(tasks['course_module_id']))
        metric_values = {}
        for module_id in modules:
            # if self.data_tables['course_module'][self.data_tables['course_module']['id'] == module_id]['is_advanced'].iloc[0]:
            #     continue
            tasks_module = tasks[tasks['course_module_id'] == module_id]
            users = np.unique(np.array(tasks_module['user_id']))
            avg_tries_for_user = []
            for user in users:
                user_tasks = tasks_module[(tasks_module['user_id'] == user) & (tasks_module['tries_count'] > 0)]
                all_tries = np.array(user_tasks['tries_count']).sum()
                if len(user_tasks) != 0:
                    avg_tries_for_user.append(all_tries / len(user_tasks))
            metric_values[module_id] = round(np.mean(avg_tries_for_user), 2)
        return pd.DataFrame({'module_id' : metric_values.keys(), self.metric_name : metric_values.values()})
# class ModuleMeanTries(Metric):
#     def evaluate(self, course_id=None):
#         tasks = self.data_tables['user_element_progress'][(self.data_tables['user_element_progress']["course_element_type"] == "task") &
#                                                 (self.data_tables['user_element_progress']["achieve_reason"] != "transferred")]
#         modules = np.unique(np.array(tasks['course_module_id']))
#         metric_values = {}
#         for module_id in modules:
#             if self.data_tables['course_module'][self.data_tables['course_module']['id'] == module_id]['is_advanced'].iloc[0]:
#                 continue
#             tasks_module = tasks[tasks['course_module_id'] == module_id]
#             users = np.unique(np.array(tasks_module['user_id']))
#             avg_tries_for_user = []
#             for user in users:
#                 user_tasks = tasks_module[(tasks_module['user_id'] == user) & (tasks_module['tries_count'] > 0)]
#                 all_tries = np.array(user_tasks['tries_count']).sum()
#                 if len(user_tasks) != 0:
#                     avg_tries_for_user.append(all_tries / len(user_tasks))
#             metric_values[module_id] = round(np.mean(avg_tries_for_user), 2)
#         return pd.DataFrame({'module_id' : metric_values.keys(), self.metric_name : metric_values.values()})