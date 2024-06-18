import pandas as pd
import json
import os
from default_metrics import default_element_metrics, default_module_metrics


dir = os.path.join('courses', 'machine_learning')
data_tables = {file[:-4] : pd.read_csv(os.path.join(dir, file)) for file
                in os.listdir(dir) if file.endswith('.csv')}


for metric in default_element_metrics:
    metric.data_tables = data_tables
for metric in default_module_metrics:
    metric.data_tables = data_tables

class ReportDataCreator:
    def __init__(self, course_id = None,
                 data_tables = None,
                 element_metrics_list = [],
                 module_metrics_list = []):
        self.element_metrics_list = element_metrics_list
        self.course_id = course_id
        self.module_metrics_list = module_metrics_list
        self.data_tables = data_tables
        self.course_module = data_tables['course_module'].copy()
        self.course_module = self.course_module[
            self.course_module.course_id == course_id
        ]
        self.course_element = data_tables['course_element'].copy()
        self.element_metrics_info = {metric.metric_name : {'threshold' : metric.threshold,
                                                           'description' : metric.description
                                                           }
                                    for metric in element_metrics_list}
        self.module_metrics_info = {metric.metric_name : {'threshold' : metric.threshold,
                                                           'description' : metric.description
                                                           }
                                    for metric in module_metrics_list}
        self.element_metrics = None
        self.module_metrics = None


    def save_metrics_info(self, path='', level='element'):
        if not path.endswith('.json'):
            raise ValueError('Указанный файл не является .json файлом')
        dir = os.path.dirname(path)
        if not os.path.isdir(dir):
            os.mkdir(dir)
        if level == 'element':
            with open(path, 'w') as f:
                json.dump(self.element_metrics_info, f)
        if level == 'module':
            with open(path, 'w') as f:
                json.dump(self.module_metrics_info, f)
        return self


    def calculate_metrics(self, level='element'):
        if level == 'element':
            output = pd.DataFrame(columns=['element_id'])
            for metric in self.element_metrics_list:
                output = pd.merge(output, metric.evaluate(course_id=self.course_id), 
                                  on='element_id', how='outer')
            self.element_metrics = output.copy()
        if level == 'module':
            output = pd.DataFrame(columns=['module_id'])
            for metric in self.module_metrics_list:
                output = pd.merge(output, metric.evaluate(course_id=self.course_id), 
                                  on='module_id', how='outer')
            self.module_metrics = output.copy()
        return self


    def save_metrics(self, savedir='', level='element'):
        if level == 'element':
            if self.element_metrics is None:
                self.calculate_metrics(level='element')
            for module in self.course_module.id:
                current_module_tasks = self.course_element[
                    self.course_element.module_id == module].sort_values(by='position')['element_id']
                module_table = pd.merge(current_module_tasks, self.element_metrics, on='element_id')
                module_table.to_csv(os.path.join(savedir, f'{module}.csv'), index=False)
        if level == 'module':
            if self.module_metrics is None:
                self.calculate_metrics(level='module')
            self.module_metrics.to_csv(os.path.join(savedir, 'module_metrics.csv'), index=False)
        return self


    def save_course_data(self, savedir=''):
        savedir = os.path.join(savedir, str(self.course_id))
        self.save_metrics_info(path=os.path.join(savedir, 'element_metrics_info.json'),
                               level='element')
        self.save_metrics_info(path=os.path.join(savedir, 'module_metrics_info.json'), 
                               level='module')
        if len(self.element_metrics_list) != 0:
            self.save_metrics(savedir=savedir, level='element')
        if len(self.module_metrics_list) != 0:
            self.save_metrics(savedir=savedir, level='module')
        return self


rep_cr = ReportDataCreator(course_id=1316,
                            data_tables=data_tables,
                            element_metrics_list=default_element_metrics,
                            module_metrics_list=default_module_metrics)

rep_cr.save_course_data()