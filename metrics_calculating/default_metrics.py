from metrics import *

# Метрики сложности задач:
mean_tries = MeanTriesCount(metric_name='Среднее число попыток',
                            parameters={'outlier' : 65}, threshold=[4, 1])
mean_tries.description = f'Среднее число попыток пользователя на данной задаче. Удаляются выбросы (число попыток > {mean_tries.parameters["outlier"]}).'

skips_percentage = SkipsPercentage(metric_name='Доля пропусков',
                            parameters={}, threshold=[0.2, 0])
skips_percentage.description = 'Доля пользователей, пропустивших задачу, среди тех, кому она открыта. Задача пропущена, если пользователь решил задачу, расположенную позже в модуле, раньше текущей (или не решил текущую задачу).'

lost_percentage =  LostPercentage(metric_name='Доля нерешивших среди продолжающих',
                            parameters={}, threshold =[0.15,0])
lost_percentage.description = 'Доля пользователей, не решивших задачу, среди решивших предыдущую. Не определена для задач после видео или текстовых материалов.'

guessed_percentage = GuessedPercentage(metric_name='Доля угадываний',
                                parameters={}, threshold=[0.2,0])
guessed_percentage.description = 'Доля пользователей, совершивших хотя бы 5 попыток с разницей менее 10 секунд между соседними попытками.'

solved_percentage = SolvedPercentage(metric_name='Доля решивших',
                                    parameters={}, threshold=[0.88, 1])
solved_percentage.description = 'Доля пользователей, решивших задачу, среди совершивших хотя бы 1 попытку.'

mean_time = TaskTime(metric_name = 'Среднее время дорешки', 
                    parameters={'metric' : 'mean'}, threshold=[7.5,0])
mean_time.description = 'Среднее суммарное время (в минутах), потраченное пользователем на данную задачу. Время на первую попытку не учитывается. Интервалы между попытками длиннее 40 минут не учитываются.'

n_tries = NTries(metric_name='Доля решивших за много попыток',
                 parameters={'N' : 7}, threshold=[0.1, 0])
n_tries.description = f'Доля пользователей, решивших задачу не менее чем с {n_tries.parameters["N"]} попытки'

default_element_metrics = [solved_percentage, mean_tries, guessed_percentage, n_tries, mean_time, lost_percentage, skips_percentage]

# Метрики сложности модуля:

module_user_count = ModuleUserCount(metric_name='Число учеников',
                                    threshold=[])
module_user_count.description = 'Количество учеников, совершивших хотя бы одну попытку в модуле.'

module_achieved_percentage = ModuleAchievedPercentage(metric_name='Доля зачетов',
                                                      threshold=[0.5, 1])
module_achieved_percentage.description = '% учеников, получивших зачет по данному модулю.'

module_time = ModuleTime(metric_name='Среднее время',
                         parameters={'max_timedelta_min' : 40}, threshold = [80, 0])
module_time.description = 'Среднее время в минутах, затраченное пользователем на модуль. Считается как сумма временных интервалов между последовательными действиями пользователя в модуле. Интервалы длиннее 40 минут не учитываются.'

module_tries = ModuleMeanTries(metric_name='Среднее число попыток',
                               threshold=[4, 1]
)
module_tries.description = f'Среднее число попыток пользователя в данном модуле.'
default_module_metrics = [module_user_count, module_achieved_percentage, module_time, module_tries]




                    
