import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import List, Dict, Union, Tuple
from abc import ABC, abstractmethod
from tqdm import tqdm
from datetime import timedelta


class Metric(ABC):
    def __init__(
        self,
        metric_name: str,
        threshold,
        parameters: Dict[str, float],
        data_tables: Union[List[DataFrame], Dict[str, DataFrame]],
    ) -> None:
        self.metric_name = metric_name
        self.threshold = threshold
        self.parameters = parameters
        self.data_tables = data_tables

    @abstractmethod
    def evaluate(
        self, *args, **kwargs
    ):  # -> DataFrame({'id': int, self.metric_name: float}):
        pass


class MetricsCalculator:
    def __init__(
        self, metrics_list: List[Metric], metrics_args: List[Tuple[List, Dict]]
    ):
        self.metrics_list = metrics_list
        self.metrics_args = metrics_args

    def calculate_metrics(self):
        args, kwargs = self.metrics_args[0]
        results = self.metrics_list[0].evaluate(*args, **kwargs)
        for metric, metric_args in zip(self.metrics_list[1:], self.metrics_args[1:]):
            args, kwargs = metric_args
            results = pd.merge(
                results, metric.evaluate(*args, **kwargs), on="id", how="outer"
            )
        return results


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

        skips_percentage = skips_count / user_count * 100
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
        lost_percentage = lost_count / solved_prev_count * 100

        return pd.DataFrame(
            {
                "element_id": lost_percentage.index,
                "lost_percentage": lost_percentage.values,
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

        for group_id, group in tqdm(solution_log.groupby("element_progress_id")):
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
        guessed_percentage = guessed_count / tried_count * 100
        return pd.DataFrame(
            {
                "element_id": guessed_percentage.index,
                "guessed_percentage": guessed_percentage.values,
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
                self.metric_name: grouped["is_achieved"] / grouped["user_id"] * 100,
            }
        )
        return solved_percentage


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
                self.metric_name: agg_df["is_achieved"] / agg_df["user_id"] * 100,
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

        for name, group in tqdm(actions.groupby(["course_module_id", "user_id"])):
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


### EXAMPLE

# mtc = MeanTriesCount(metric_name='mean_tries_count', data_tables={'user_element_progress': user_element_progress},
#                     parameters={'outlier': 65}, threshold=4)
# std = TriesStd(metric_name='tries_std', data_tables={'user_element_progress': user_element_progress},
#                     parameters={'outlier': 65}, threshold=5)
# skips = SkipsPercentage(metric_name='skips_percentage', data_tables={'user_element_progress': user_element_progress, 'course_element': course_element},
#                     parameters={}, threshold=0.2)
# lost = LostPercentage(metric_name='lost_percentage', data_tables={'user_element_progress': user_element_progress, 'course_element': course_element},
#                     parameters={}, threshold=0.15)
# guess = GuessedPercentage(metric_name='guessed_percentage', data_tables={'user_element_progress': user_element_progress, 'solution_log': solution_log},
#                     parameters={'course_id': course_id}, threshold=0.2)
# sp = SolvedPercentage('solved_percentage', threshold=0.88, parameters={},data_tables={'user_element_progress': user_element_progress})
# user_count_metric = ModuleUserCount(metric_name='user_count', data_tables=data_tables, threshold=None, parameters=None)
# achieved_metric = ModuleAchievedPercentage(metric_name='achieved_percentage', data_tables=data_tables, threshold=None, parameters=None)
# module_time_metric = ModuleTime(metric_name='module_time', data_tables=data_tables, threshold=None, parameters={'max_timedelta_min': 40})
