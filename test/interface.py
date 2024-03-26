import pandas as pd
import numpy as np
import os


class MetricsCalculator:
    def __init__(self, course_folder):
        if os.path.dirname(course_folder) == '':
            course_folder = os.path.join(os.path.dirname(__file__),
                                         course_folder)
        self.course_element = pd.read_csv(os.path.join(course_folder,
                                          'course_element.csv'))
        self.course_module = pd.read_csv(os.path.join(course_folder,
                                                      'course_module.csv'))
        self.course = pd.read_csv(os.path.join(course_folder,
                                               'course.csv'))
        self.solution_log = pd.read_csv(os.path.join(course_folder, 
                                                     'solution_log.csv'))
        self.user_course_progress = pd.read_csv(os.path.join(course_folder,
                                                'user_course_progress.csv'))
        self.user_element_progress = pd.read_csv(os.path.join(course_folder, 
                                                 'user_element_progress.csv'))
        self.user_module_progress = pd.read_csv(os.path.join(course_folder,  
                                                'user_module_progress.csv'))


    def modules_list(self):
        return np.array(self.course_module.id)

    
    def tasks_list(self, module_id):
        course_element = self.course_element[['id', 'module_id',
                                                  'element_type', 'element_id']]
        course_element = course_element[course_element.module_id ==
                                             module_id]
        course_element = course_element[course_element.element_type == 'task']
        return np.array(course_element.element_id)

        
    def number_of_tries(self, module_id, metric='median'):
        if metric == 'mean':
            metric = np.mean
        else:
            metric = np.median
            
        user_current_element_progress = self.user_element_progress[
            self.user_element_progress.course_module_id == module_id]
        user_current_element_progress = user_current_element_progress[
            user_current_element_progress.course_element_type == 'task']
        user_current_element_progress = user_current_element_progress[
            user_current_element_progress.is_achieved == True]
        user_current_element_progress = user_current_element_progress[
            user_current_element_progress.achieve_reason == 'solved']
    
        current_tasks = list(set(
            user_current_element_progress.course_element_id))
        task_metric = []
        for element_id in current_tasks:
            task_metric.append(metric(user_current_element_progress[
                user_current_element_progress.course_element_id == element_id
                ].tries_count))
        return pd.DataFrame(data=task_metric, index=[current_tasks], dtype=float)