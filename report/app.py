import pandas as pd
import numpy as np
from datetime import datetime
import os
import networkx
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash import callback
from dash.dependencies import Input, Output


COURSE_ID = 1316


#---------------------------------------------------------------
# VERY useful functions

def extract_dates():
    start = datetime.fromisoformat(str(course[course.id == COURSE_ID].date_start.iloc[0]))
    close = datetime.fromisoformat(str(course[course.id == COURSE_ID].close_date.iloc[0]))
    now = datetime.now()
    message = '(Курс еще идет!)' if close > now else ''
    dates = html.Div([
        html.Div(f'Дата открытия курса: {start}'[:-8]),
        html.Div(f'Дата закрытия курса: {close} {message}', style={'color' : 'red' if now < close else 'black'})
    ], className='course-dates')
    return dates


def create_one_factoid(title, value): # создает один фактоид по шаблону
    factoid = html.Section([
        html.H4(f'{title}', className='factoid-header'),
        html.P(f'{value}', className='factoid-text')
    ], className='factoid')
    return factoid


def create_course_factoids(): # вычисляет фактоиды курса и создает их
    # зарегистрировано учеников:
    registered = user_course_progress.shape[0]
    # начало обучение (зачет по любому элементу без автогрейда):
    started = len(user_element_progress[user_element_progress.achieve_reason \
                                        == 'solved'].user_id.unique())
    # активно учеников (зачет по любому модулю без автогрейда)
    active = len(user_module_progress[user_module_progress.achieve_reason == \
                                      'solved'].course_progress_id.unique())
    # число сертификатов (зачет по всем ordinary модулям 1-го уровня):
    ordinary_first_level_modules = list(course_module[(course_module.type == 'ordinary') &
                                                      (course_module.level == 1)].id)
    achieved_modules = user_module_progress[(user_module_progress.is_achieved) &
                                            (user_module_progress.course_module_id \
                                             .isin(ordinary_first_level_modules))]
    n_certificates = 0
    for id in achieved_modules.course_progress_id.unique():
        if (achieved_modules.course_progress_id == id).sum() == len(ordinary_first_level_modules):
            n_certificates += 1
    # создаем фактоиды и объединяем их вместе:
    factoids = html.Div([
        create_one_factoid('Зарегистрировано:', registered),
        create_one_factoid('Начало обучение:', started),
        create_one_factoid('Активных учеников:', active),
        create_one_factoid('Получено сертификатов:', n_certificates)
    ], className='flex')
    return factoids


def colors(value, interval, func = lambda x: x):
    '''
    interval : list, possible forms:
    1. [threshold, min_or_max_value] if we want metric to be less/greater
    then threshold
    2. [threshold_1, threshold_2, center] if we want metric to be between two values,
    center - parameter which determines walue to color white
    '''
    if value == '':
        return 'rgb(159,0,255)'
    if len(interval) == 2:
        threshold = interval[0]
        border = interval[1]
        if value == threshold:
            return f'rgb(255,0,0)'
        if value > threshold > border:
            return f'rgb(255,0,0)'
        if value < threshold < border:
            return f'rgb(255,0,0)'
        else:
            v = int(255 * func(abs(value - threshold)) / func(abs(border - threshold)))
            return f'rgb(255,{v},{v})'
    if len(interval) == 3:
        print('Такой случай пока не реализован')
        return 'pink'


def create_course_graph(metric_name): # граф курса
    graph = networkx.Graph().to_directed_class()
    graph = networkx.from_pandas_edgelist(course_graph, source='from_module_id', target='to_module_id', create_using=graph)
    positions = networkx.layout.planar_layout(graph)
    for node in graph.nodes:
        graph.nodes[node]['position'] = list(positions[node])
        graph.nodes[node]['id'] = node
        graph.nodes[node]['type'] = course_module[course_module.id == node].type.iloc[0]
    nodes = go.Scatter(x=[], y=[], text=[], mode='markers+text', textposition='middle center', 
                       marker={'size' : 43,  'symbol' : [], 'color' : [], 'opacity' : 1, 'line' : {'width' : 1, 'color' : 'black'}})
    for node in graph.nodes:
        x, y = graph.nodes[node]['position']
        nodes['x'] += x,
        nodes['y'] += y,
        text = graph.nodes[node]['id']
        nodes['text'] += f'<a href="#module_{12953}" class="module-link"> {text} </a>',
        nodes['marker']['symbol'] += 'circle' if graph.nodes[node]['type'] == 'ordinary' else 'pentagon', 
        value = float(module_metrics_df[module_metrics_df.module_id == graph.nodes[node]['id']][f'{metric_name}'].iloc[0])
        nodes['marker']['color'] += colors(value, module_metrics_thresholds[metric_name]),
    lines = []
    for e in graph.edges:
        x_from, y_from = graph.nodes[e[0]]['position']
        x_to, y_to = graph.nodes[e[1]]['position']
        line = go.Scatter(x=(x_from, x_to, None), y=(y_from, y_to, None),
                        mode='lines',
                        marker=dict(color='black'),
                        line_shape='spline')
        lines.append(line)
    data = lines
    data.extend([nodes])
    fig = go.Figure(data = data, 
                    layout = go.Layout(height=900,
                        plot_bgcolor='white',
                        showlegend=False,
                        hovermode='closest',
                        xaxis={'showgrid' : False, 'zeroline' : False, 'showticklabels' : False},
                        yaxis={'showgrid' : False, 'zeroline' : False, 'showticklabels' : False},
                        annotations = [dict(
                            ax = (graph.nodes[e[0]]['position'][0] + graph.nodes[e[1]]['position'][0]) / 2,
                            ay = (graph.nodes[e[0]]['position'][1] + graph.nodes[e[1]]['position'][1]) / 2, axref='x', ayref='y',
                            x = (graph.nodes[e[1]]['position'][0] * 3 + graph.nodes[e[0]]['position'][0]) / 4,
                            y = (graph.nodes[e[1]]['position'][1] * 3 + graph.nodes[e[0]]['position'][1]) / 4, xref='x', yref='y',
                            showarrow=True,
                            arrowhead=3,
                            arrowsize=4,
                            arrowwidth=1,
                            opacity=1
                    ) for e in graph.edges]
                    ))
    return fig


def create_course_table(metric_name): # легенда к графу курса
    module_ids = list(module_metrics_df.module_id)
    metric_values = list(module_metrics_df[f'{metric_name}'].round(2))
    cell_colors = [['lightgrey'] * len(module_ids), [colors(float(module_metrics_df[module_metrics_df.module_id == id][f'{metric_name}'].iloc[0]), 
                                                       module_metrics_thresholds[metric_name])
                                                for id in module_ids]]
    fig = go.Figure(data=[go.Table(
                                    header = dict(
                                        values=['module_id', metric_name],
                                        fill_color = 'lightgrey',
                                        line_color='black'
                                    ),
                                    cells = dict(values=[module_ids, metric_values],
                                                fill_color = cell_colors,
                                                line_color='black',
                                                ),
                                    columnwidth = [0.1] + [0.1]
                                    )
                        ], layout=go.Layout(height = 500, width = 400)
                    )
    return fig


def create_course_graph_and_table(): # соединяет граф курса, табличку к нему и выпадающий список
    dropdown = html.Div(
        dcc.Dropdown(
            list(module_metrics_df.columns[1:]),
            'n_students',
            id = 'graph_with_table_metric'
        ), className='dropdown-graph'
    )
    graph = html.Div(
        dcc.Graph(id='graph'),
        className='graph'
    )
    table = html.Div(
        dcc.Graph(id='graph_table'),
        className='graph-table'
    )
    graph_with_table = html.Div([
        graph,
        table
    ], className='graph-with-table')
    
    dropdown_graph_table = html.Div([
        dropdown,
        graph_with_table
    ], className='dropdown-graph-table')
    @callback(Output('graph', 'figure'),
            Output('graph_table', 'figure'),
            Input('graph_with_table_metric', 'value')          
            )
    def update_graph(metric_name):
        fig_graph = create_course_graph(metric_name)
        fig_table = create_course_table(metric_name)
        return fig_graph, fig_table
    return dropdown_graph_table


def create_module_table(module_id): #создает табличку для модуля
    current_module_ce = course_element[course_element.module_id == module_id]
    merged_with_metrics = pd.merge(current_module_ce, metrics_dfs[f'{module_id}'], how='outer', on='element_id').sort_values('position')
    merged_with_metrics = merged_with_metrics.set_index(merged_with_metrics.position)
    merged_with_metrics = merged_with_metrics.transpose().fillna('')
    element_types = list(merged_with_metrics.loc['element_type', :])
    header = [''] + [f'{id}' for id in merged_with_metrics.loc['element_id', :]]
    merged_with_metrics = merged_with_metrics.drop(['id', 'module_id', 'element_type','is_advanced', 'max_tries', 'score', 'position', 'element_id'])
    data = [list(merged_with_metrics.index),]
    fill_color = [['lightgrey'] * merged_with_metrics.shape[0]]
    line_color = [['black'] * merged_with_metrics.shape[0]]
    for i in range(len(element_types)):
        if element_types[i] != 'task':
            data += [[''] * merged_with_metrics.shape[0]]
            data[i + 1][merged_with_metrics.shape[0] // 2] = element_types[i]
            fill_color += [['rgb(159,0,255)'] * merged_with_metrics.shape[0]]
            line_color += [['rgb(159,0,255)'] * merged_with_metrics.shape[0]] 
        else:
            data += [list(pd.to_numeric(merged_with_metrics.iloc[0:, i]).round(2)),]
            fill_color += [[colors(merged_with_metrics.loc[metric, i + 1], metrics_thresholds[metric]) 
                            for metric in merged_with_metrics.index]]
            line_color += [['black'] * merged_with_metrics.shape[0]]
    fig = go.Figure(data=[go.Table(
        header = dict(
            values = header,
            align = 'center',
            fill_color = 'lightgrey',
            line_color = 'black',),
        cells = dict(
            values = data,
            fill_color = fill_color,
            line_color = 'black',
            line_width = 1
        ),
        columnwidth = [2] + [1] * (len(fill_color) - 1)
    )],
    layout = go.Layout(height = 500, width = len(header) * (2000 // 32)))
    return fig


def create_module_factoids(module_id): # создает фактоиды модуля
    metrics = module_metrics_df[module_metrics_df.module_id == module_id]
    factoids = html.Div([
        create_one_factoid('Число зачетов', metrics.loc[:,'n_students'].iloc[0]),
        create_one_factoid('Процент зачетов', metrics.loc[:,'perc_achieved'].iloc[0]),
    ], className='flex')
    return factoids


def create_module_report(module_id):
    factoids = create_module_factoids(module_id)
    table = dcc.Graph(figure=create_module_table(module_id), className='module-table')
    module_report = html.Div([
        html.H3(f'Название модуля (ID = {module_id})'),
        factoids,
        table
    ], className='module-report', id=f'module_{module_id}')
    return module_report


#----------------------------------------------------------------
# Creating app
app = dash.Dash(__name__)


#-----------------------------------------------------------------
# Connecting data
course = pd.read_csv('courses/machine_learning/course.csv')
course_graph = pd.read_csv('courses/machine_learning/course_graph.csv')
course_graph = course_graph[course_graph.course_id == COURSE_ID]
course_module = pd.read_csv('courses/machine_learning/course_module.csv')
course_module = course_module[course_module.course_id == COURSE_ID]
course_element = pd.read_csv('courses/machine_learning/course_element.csv')
user_course_progress = pd.read_csv('courses/machine_learning/user_course_progress.csv')
user_course_progress = user_course_progress[user_course_progress.course_id == COURSE_ID]
user_module_progress = pd.read_csv('courses/machine_learning/user_module_progress.csv')
user_module_progress = user_module_progress[user_module_progress.course_id == COURSE_ID]
user_element_progress = pd.read_csv('courses/machine_learning/user_element_progress.csv')
user_element_progress = user_element_progress[user_element_progress.course_id == COURSE_ID]
metrics_dfs = {file[:-5] : pd.read_excel(os.path.join(str(COURSE_ID), file)) if file.endswith('xlsx')
       else None for file in os.listdir(f'{COURSE_ID}/')}
with open(os.path.join(str(COURSE_ID), 'thresholds.txt'), 'r') as f:
    thresholds = f.read()
thresholds = [float(i) for i in thresholds.split()]
thresholds[:3] = [10, 1, 0.9]
metrics_thresholds = {metric_name : [ thresholds[i], 0] for i, metric_name 
                      in enumerate(metrics_dfs['12953'].columns[1:])}
metrics_thresholds['solved_perc'] = [metrics_thresholds['solved_perc'][0], 1]
################################################################
# Эта часть тут временно и позже будет заменена на выгрузку из 
# таблички, тут считаются всякие метрики модуля
n_students = []
n_achieved = []
ids = []
for id in course_module.id:
    ids.append(id)
    n_students.append(len(user_module_progress[user_module_progress.course_module_id == id]))
    n_achieved.append(len(user_module_progress[(user_module_progress.course_module_id == id) & 
                                               (user_module_progress.is_achieved)]))
n_students = np.array(n_students)
n_achieved = np.array(n_achieved)
perc_achieved = n_achieved / n_students
module_metrics_df = pd.DataFrame(data = {'module_id' : ids,'n_students' : n_students, 'perc_achieved' : perc_achieved})
module_metrics_df['perc_achieved'] = module_metrics_df['perc_achieved'].round(2)
module_metrics_thresholds = {metric_name : [0, np.max(module_metrics_df[metric_name])] 
                             for metric_name in module_metrics_df.columns[1:]}
################################################################
#---------------------------------------------------------------------
# Creating all elements for layout
header = html.H1(f'Введение в машинное обучение (ID = {COURSE_ID})')
dates = extract_dates()
course_factoids = create_course_factoids()
course_graph_and_table = create_course_graph_and_table()
modules_reports = html.Div([create_module_report(id) for id in course_module.id])

#-----------------------------------
# App layout
app.layout = html.Div([
    header,
    dates,
    course_factoids,
    html.H2('Карта курса'),
    html.Div([
    course_graph_and_table,
    ], className='vspace'),
    html.H2('Анализ каждого модуля'),
    modules_reports
])
if __name__ == '__main__':
    app.run_server(debug=True)
