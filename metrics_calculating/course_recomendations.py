from metrics import *


def get_data(course_id, metrics_list, course_info):
    """course_id -- номер запуска, metrcis_list -- список метрик, course_info -- словарь с информацией по всем запускам курса"""
    element_info = {}
    course_module = course_info['course_module']
    course_element = course_info['course_element']
    output = pd.DataFrame(columns=['element_id'])
    elements = np.unique(np.array(course_element['element_id']))
    for element in elements:
        element_info[element] = {}
    for metric in metrics_list:
        output = metric.evaluate(course_id=course_id)
        if metric.metric_name != 'solved_percentage':
            for (x, y) in zip(output['element_id'], output[metric.metric_name]):
                element_info[x][metric.metric_name] = (y, y > metric.threshold)
        else:
            for (x, y) in zip(output['element_id'], output[metric.metric_name]):
                element_info[x][metric.metric_name] = (y, y < metric.threshold)
    return element_info

def init_metrics(course_info):
    """course_info -- словарь с информацией по всем запускам курса"""
    Ntries = NTries(metric_name='N_tries', threshold=0.1, parameters={'N' : 7}, data_tables=course_info)
    mt = TaskTime(metric_name='mean_time', data_tables=course_info, parameters={'metric' : 'mean'}, threshold=7.5)
    mtc = MeanTriesCount(metric_name='mean_tries_count', data_tables={'user_element_progress': course_info['user_element_progress']},
                     parameters={'outlier': 65}, threshold=4)
    skips = SkipsPercentage(metric_name='skips_percentage', data_tables={'user_element_progress': course_info['user_element_progress'], 'course_element': course_info['course_element']},
                        parameters={}, threshold=0.2)
    lost = LostPercentage(metric_name='lost_percentage', data_tables={'user_element_progress': course_info['user_element_progress'], 'course_element': course_info['course_element']},
                        parameters={}, threshold=0.15)
    guess = GuessedPercentage(metric_name='guessed_percentage', data_tables={'user_element_progress': course_info['user_element_progress'], 'solution_log': course_info['solution_log']},
                        parameters={}, threshold=0.2)
    sp = SolvedPercentage('solved_percentage', threshold=0.88, parameters={}, data_tables={'user_element_progress': course_info['user_element_progress']})
    return [Ntries, mt, mtc, skips, lost, guess, sp]
used_metrics = ['N_tries', 'mean_time', 'mean_tries_count', 'skips_percentage',
                'lost_percentage', 'guessed_percentage', 'solved_perentage']
def crit1(data):
    elements = []
    for key in data.keys():
        if 'mean_time' in data[key].keys() and 'mean_tries_count' in data[key].keys() and 'guessed_percentage' in data[key].keys():
            if data[key]['mean_time'][1] and data[key]['mean_tries_count'][1] and not data[key]['guessed_percentage'][1]:
                elements.append(key)
    return elements

def crit2(data):
    elements = []
    for key in data.keys():
        if 'solved_percentage' in data[key].keys() and 'skips_percentage' in data[key].keys():
            if data[key]['solved_percentage'][0] >= 0.94 and data[key]['skips_percentage'][1]:
                elements.append(key)
    return elements

def crit3(data):
    elements = []
    for key in data.keys():
        count = 0
        for metric in used_metrics:
            if metric in data[key].keys():
                count += data[key][metric][1]
        if count >= 4:
            elements.append(key)
    return elements

def enum(elements):
    res = ''
    n = len(elements)
    for i in range(n):
        if i != n-1 and i != n-2:
            res += str(elements[i]) + ', '
        elif i == n-2:
            res += str(elements[i]) + ' и '
        else:
            res += str(elements[i])
    return res

def generate_recomendations(data):
    """data -- словарь, ключ -- номер задачи, значение -- словарь, где ключ -- название метрики, значение -- пара (значение метрики, флаг, является ли оно критическим)"""
    return [enum(crit1(data)), enum(crit2(data)), enum(crit2(data))]
    # print('Задачи ' + enum(crit1(data)) + ' требуют у учеников много времени и попыток.')
    # print('Задачи ' + enum(crit2(data)) + ' часто пропускают, однако среди приступивших процент решений очень высок.')
    # print('О сложности задач ' + enum(crit3(data)) + ' сигнализирует больше половины метрик.')