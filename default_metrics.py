from metrics import *

# Метрики сложности задач:
mean_tries = MeanTriesCount(metric_name='попыток в среднем',
                            parameters={'outlier' : 65}, threshold=[4, 0])
mean_tries.description = 'описание mean_tries'

skips_percentage = SkipsPercentage(metric_name='доля пропусков',
                            parameters={}, threshold=[0.2, 0])
skips_percentage.description = 'описание skips_perc'

lost_percentage =  LostPercentage(metric_name='доля не решивших',
                            parameters={}, threshold =[0.15,0])
lost_percentage.description = 'описание lost_perc'

guessed_percentage = GuessedPercentage(metric_name='доля угадываний',
                                parameters={}, threshold=[0.2,0])
guessed_percentage.description = ' описание guessed_perc'

solved_percentage = SolvedPercentage(metric_name='доля решивших',
                                    parameters={}, threshold=[0.88, 1])
solved_percentage.description = 'описание solved_percentage'

mean_time = TaskTime(metric_name = 'среднее время', 
                    parameters={'metric' : 'mean'}, threshold=[7.5,0])
mean_time.description = 'описание mean_time'

n_tries = NTries(metric_name='доля сделавших много попыток',
                 parameters={'N' : 7}, threshold=[0.1, 0])
n_tries.description = 'описание n_tries'

default_element_metrics = [mean_tries, skips_percentage, lost_percentage, guessed_percentage,
                           solved_percentage, mean_time, n_tries]

# Метрики сложности модуля:

module_user_count = ModuleUserCount(metric_name='число учеников',
                                    threshold=[])
module_user_count.description = 'описание module_user_count'

module_achieved_percentage = ModuleAchievedPercentage(metric_name='доля зачетов',
                                                      threshold=[100, ])
module_achieved_percentage.description = 'описание module_achieved_percentage'

module_time = ModuleTime(metric_name='время на решение',
                         parameters={'max_timedelta_min' : 40}, threshold = [40, 0])
module_time.description = 'описание module_time'

default_module_metrics = [module_user_count, module_achieved_percentage, module_time]




                    
