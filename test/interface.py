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