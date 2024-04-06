class Metric:
    def __init__(self, course_element=None, course_module=None, course=None,
                 solution_log=None, user_course_progress=None,
                 user_element_progress=None, user_module_progress=None):
        
        self.course_element = course_element
        self.course_module = course_module
        self.course = course
        self.solution_log = solution_log
        self.user_course_progress = user_course_progress
        self.user_element_progress = user_element_progress
        self.user_module_progress = user_module_progress


    def evaluate(self, *args, **kwargs):
        raise NotImplementedError


    def __call__(self):
        raise NotImplementedError


class MetricsCalculator:
    def __init__(self, metrics_list):
        self.metrics_list = metrics_list


    def calculate_metrics(self):
        output = []
        for metric in self.metrics_list:
            output.append(metric())
        return output


class MeanTriesCount(Metric):
    def evaluate(self, *args, **kwargs):
        TRIES_COUNT_OUTLIER = 65
        user_element_progress = self.user_element_progress
        df = user_element_progress[user_element_progress['course_element_type'] == 'task']
        df = df[df['achieve_reason'] != 'transferred']
        df_filtered = df[df['tries_count'] < TRIES_COUNT_OUTLIER]
        df_filtered = df[df['tries_count'] > 0]
        mean_tries= df_filtered[['course_element_id', 'tries_count']].groupby(by='course_element_id').mean()['tries_count']
        return pd.DataFrame({'mean_tries_count': mean_tries.values}, index=mean_tries.index)
        

class TriesStd(Metric):
    def evaluate(self, *args, **kwargs):
        TRIES_COUNT_OUTLIER = 65
        user_element_progress = self.user_element_progress
        df = user_element_progress[user_element_progress['course_element_type'] == 'task']
        df = df[df['achieve_reason'] != 'transferred']
        df_filtered = df[df['tries_count'] < TRIES_COUNT_OUTLIER]
        tries_std = df_filtered[['course_element_id', 'tries_count']].groupby('course_element_id').std()['tries_count']
        return pd.DataFrame({'tries_std': tries_std.values}, index=tries_std.index)

class SkipsPercentage(Metric):
    def evaluate(self, *args, **kwargs):
        columns_for_skips = ['user_id', 'course_element_id', 'course_module_id', 'position', 'time_achieved']
        course_element = self.course_element
        course_element = course_element.rename(columns={'module_id': 'course_module_id', 'element_id': 'course_element_id', 'element_type': 'course_element_type'})
        user_element_progress = self.user_element_progress
        df = user_element_progress[user_element_progress['course_element_type'] == 'task']
        df = df[df['achieve_reason'] != 'transferred']
        df_for_skips = pd.merge(df, course_element, how='inner', on=['course_module_id',	'course_element_type',	'course_element_id'])[columns_for_skips]

        groups_with_skipped_col = []
        columns = ['user_id', 'course_element_id', 'skipped']
        
        for groupname, group in df_for_skips.groupby(['user_id', 'course_module_id']):
            group = self.add_skipped_column(group)[columns]
            groups_with_skipped_col.append(group)
        
        skips_data = pd.concat(groups_with_skipped_col)
        skips_count = skips_data.groupby(by='course_element_id').sum()['skipped']
        # согласно документации, запись о пользователе-элементе в user_element_progress есть только тогда, когда задача открыта пользователю
        user_count = skips_data.groupby(by='course_element_id').count()['user_id']

        skips_percentage = skips_count / user_count
        return pd.DataFrame({'skips_percentage': skips_percentage.values}, index=skips_percentage.index)

    def add_skipped_column(self, group):
        """Добавляет колонку skipped о пропуске задачи для подмножества user_element_progress с фиксированными module_id, user_id"""
        group = group.sort_values(by='position')     # для корректности на NaN
        group = group.sort_values(by='time_achieved')
        group['shift_position'] = group['position'].shift()
        group['max_solved_before'] = group['shift_position'].rolling(2, min_periods=1).max()
        # номер задачи меньше максимального номера уже решенной задачи => ее пропускали
        group['skipped'] = group['position'].le(group['max_solved_before'])
        # group['skipped'] = group['position'].le(group['position'].shift().rolling(2, min_periods=1).max())
        return group

class LostPercentage(Metric):
    def evaluate(self, *args, **kwargs):
        user_element_progress = self.user_element_progress
        df = user_element_progress[user_element_progress['course_element_type'] == 'task']
        df = df[df['achieve_reason'] != 'transferred']
        course_element = self.course_element
        course_element = course_element.rename(columns={'module_id': 'course_module_id', 'element_id': 'course_element_id', 'element_type': 'course_element_type'})
        df_for_lost = pd.merge(user_element_progress, course_element, how='inner', on=['course_module_id',	'course_element_type',	'course_element_id'])

        columns = ['user_id', 'course_module_id', 'course_element_type', 'course_element_id', 'position', 'is_achieved']
        df_for_lost = df_for_lost[columns]
        groups = []

        for groupname, group in df_for_lost.groupby(['user_id', 'course_module_id']):
            group = group.sort_values('position')
            group['prev_is_task'] = ('task' == group['course_element_type'].shift())
            group['prev_is_solved'] = group['is_achieved'].shift()
            # оставим только информацию о задачах
            group = group[group['course_element_type'] == 'task']
            # оставим только задачи, идущие в модуле сразу после других задач (а не видео/текста)
            group = group[group['prev_is_task']]
            groups.append(group)

        lost_data = pd.concat(groups)
        lost_data['is_achieved'] = lost_data['is_achieved'].fillna(False)
        lost_data['is_lost'] = (lost_data['prev_is_solved'] & (~lost_data['is_achieved']))
        counts_df = lost_data[['course_element_id', 'prev_is_solved', 'is_lost']].groupby('course_element_id').sum()
        lost_count = counts_df['is_lost']
        solved_prev_count = counts_df['prev_is_solved']
        lost_percentage = lost_count / solved_prev_count

        return pd.DataFrame({'lost_percentage': lost_percentage.values}, index=lost_percentage.index)

class GuessedPercentage(Metric):
    def evaluate(self, *args, **kwargs):
        user_element_progress = self.user_element_progress
        df = user_element_progress[user_element_progress['course_element_type'] == 'task']
        df = df[df['achieve_reason'] != 'transferred']
        df_tried = df[df['tries_count'] > 0]
        solution_log = self.solution_log
        solution_log['submission_time'] = pd.to_datetime(solution_log['submission_time'])
        
        element_progress_ids = solution_log['element_progress_id'].unique()
        # по element_progress_id записано, угадывалась ли данная задача данным пользователем
        progress_id_guessed = pd.Series(data=[False]*len(element_progress_ids), index=element_progress_ids)

        max_time_delta = pd.Timedelta(seconds=10)
        N = 3
        
        for group_id, group in tqdm(solution_log.groupby('element_progress_id')):
            if len(group) < N:
                # если N попыток не сделано, то "угадывания" точно не было
                progress_id_guessed[group_id] = False
                continue
            group = group.sort_values('submission_time')
            # group['prev_time'] = group['submission_time'].shift()
            # group['time_delta'] = group['submission_time'] - group['submission_time'].shift()
            group['is_guess'] = ((group['submission_time'] - group['submission_time'].shift())<= max_time_delta)
            group['is_N_guesses'] = group['is_guess'].rolling(N, min_periods=1).min()
            progress_id_guessed[group_id] = group['is_N_guesses'].any()

        guess_data = pd.DataFrame({'id': progress_id_guessed.index, 'guessed': progress_id_guessed.values})
        df_tried = pd.merge(df_tried, guess_data, how='left', on='id')
        # количество угадывавших
        guessed_count = df_tried[['course_element_id', 'guessed']].groupby('course_element_id').sum()['guessed']
        # количество пытавшихся решить задачу
        tried_count = df_tried[['course_element_id', 'tries_count']].groupby('course_element_id').count()['tries_count']
        guessed_percentage = guessed_count/tried_count
        return pd.DataFrame({'guessed_percentage': guessed_percentage.values}, index=guessed_percentage.index)
