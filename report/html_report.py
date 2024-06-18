from jinja2 import FileSystemLoader, Environment
import os
import pandas as pd
import json
from datetime import datetime
import networkx
import plotly.graph_objs as go
import re
import numpy as np


COURSE_ID = 1316
DIR = os.path.dirname(__file__)
COURSE_INFO_FOLDER = os.path.join(DIR, 'courses/machine_learning')
METRICS_FOLDER = os.path.join(DIR, '1316')
TEMPLATE_PATH = os.path.join(DIR, 'templates')
CSS_FILE = os.path.join(DIR, 'templates', 'style.css')
OUTPUT_FILE = os.path.join(DIR, f'report_{COURSE_ID}.html')
FAN_PATH = os.path.join(DIR, 'courses','machine_learning','fans')

def extract_dates():
    start = datetime.fromisoformat(str(course[course.id == COURSE_ID].date_start.iloc[0]))
    close = datetime.fromisoformat(str(course[course.id == COURSE_ID].close_date.iloc[0]))
    now = datetime.now()
    message = '(Курс еще идет!)' if close > now else ''
    return str(start)[:-8], str(close)[:-8] + message


def create_course_factoids():
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
    factoids = [factoid_temp.render(title='Зарегистрировано', value=registered),
                factoid_temp.render(title='Начало обучение', value=started),
                factoid_temp.render(title='Активно учеников', value=active),
                factoid_temp.render(title='Выдано сертификатов', value=n_certificates)]
    return factoids


def colors(value, interval, func = lambda x: x):
    '''
    interval : list, possible forms:
    1. [threshold, min_or_max_value] if we want metric to be less/greater
    then threshold
    2. [threshold_1, threshold_2, center] if we want metric to be between two values,
    center - parameter which determines walue to color white
    '''
    if value == '' or np.isnan(value):
        return 'lightgrey'
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


def create_course_graph():
    graph = networkx.Graph().to_directed_class()
    graph = networkx.from_pandas_edgelist(course_graph, source='from_module_id', target='to_module_id', create_using=graph)
    positions = json.loads(course_graph_layout.content.iloc[0])['module_positions'] 
    for node in graph.nodes:
        x, y = positions[str(node)]['x'], -positions[str(node)]['y']
        graph.nodes[node]['position'] = {'x' : x, 'y' : y}
        graph.nodes[node]['id'] = node
        graph.nodes[node]['type'] = course_module[course_module.id == node].type.iloc[0]
    sources = set(course_graph.from_module_id.unique())
    destinitions = set(course_graph.to_module_id.unique())
    from_zero_node = list(sources.difference(destinitions))
    graph.add_node(0)
    graph.nodes[0]['position'] = {'x' : 0, 'y' : 0}
    graph.nodes[0]['id'] = ''
    graph.nodes[0]['type'] = 'graph_source'
    for node in from_zero_node:
        graph.add_edge(0, node)

    def create_graph_lines(graph):
        lines = []
        for e in graph.edges:
            x_from, y_from = graph.nodes[e[0]]['position']['x'], graph.nodes[e[0]]['position']['y']
            x_to, y_to = graph.nodes[e[1]]['position']['x'],  graph.nodes[e[1]]['position']['y']
            line = go.Scatter(x=(x_from, x_to, None), y=(y_from, y_to, None), hoverinfo='none',
                            #mode='lines',
                            marker=dict(symbol="arrow-bar-up", color='rgb(42, 63, 95)', angleref="previous"),
                            line_shape='spline')
            lines.append(line)
        return lines


    def create_graph_nodes(graph, metric_name):
        nodes = go.Scatter(x=[], y=[], text=[], mode='markers+text', textposition='middle center', hoverinfo='none', customdata=[],
                            marker={'size' : 50,  'symbol' : [], 'color' : [], 'opacity' : 1, 'line' : {'width' : 1, 'color' : 'black'}})
        for node in graph.nodes:
            x = graph.nodes[node]['position']['x']
            y = graph.nodes[node]['position']['y']
            nodes['x'] += x,
            nodes['y'] += y,
            text = graph.nodes[node]['id']
            nodes['text'] += f'{text}',
            nodes['marker']['symbol'] += 'pentagon' if graph.nodes[node]['type'] == 'advanced' else 'circle', 
            if graph.nodes[node]['type'] == 'graph_source':
                color = 'lightgrey'
            else:
                value = float(module_metrics_df[module_metrics_df.module_id == graph.nodes[node]['id']][f'{metric_name}'].iloc[0])
                color = colors(value, module_metrics_thresholds[metric_name])
            nodes['marker']['color'] += color,
        return nodes

    def calculate_arrow_pos(e0, e1):
        if e1['type'] == 'ordinary': 
            r = 48
        else:
            r = 40
        
        x0, y0 = e0['position']['x'], e0['position']['y']
        x1, y1 = e1['position']['x'], e1['position']['y']
        l = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        vx = (x1 - x0) * (1 - r/l)
        vy = (y1 - y0) * (1 - r/l)
        x = x0 + vx
        y = y0 + vy
        ax = x1 - vx
        ay = y1 - vy 
        return ax, ay, x, y


    fig = go.Figure()
    lines = create_graph_lines(graph)
    n_lines = len(lines)
    for line in lines:
        fig.add_trace(line)
    fig.add_trace(create_graph_nodes(graph, 'Доля зачетов'))
    fig.add_trace(create_graph_nodes(graph, 'Среднее время'))
    fig.add_trace(create_graph_nodes(graph, 'Среднее число попыток'))
    fig.update_traces(visible=False, selector=-1)
    fig.update_traces(visible=False, selector=-2)
    fig.update_layout(updatemenus = [{
        'active' : 0,
        'showactive' : True,
        'buttons' : [
            {'label' : 'Доля зачетов',
            'method' : 'update',
            'args' : [{'visible' : [True] * n_lines + [True, False, False]}]},
            {'label' : 'Среднее время',
            'method' : 'update',
            'args' : [{'visible' : [True] * n_lines + [False, True, False]}]},
            {'label' : 'Доля зачетов',
            'method' : 'update',
            'args' : [{'visible' : [True] * n_lines + [False, False, True]}]}
        ],
        'pad' : {'r' : 10, 't' : 0},
        'x' : 0,
        'xanchor' : 'left',
        'y' : 1,
        'yanchor' : 'top',
        'font' : {'size' : 12}

        # 'x' : 0.01,
        # 'y' : 1,

    }])


    arrows = [calculate_arrow_pos(graph.nodes[e[0]], graph.nodes[e[1]]) for e in graph.edges]
    fig.update_layout(dict(height=530, width=530,
                        plot_bgcolor='white',
                        showlegend=False,
                        hovermode='closest',
                        xaxis={'showgrid' : False, 'zeroline' : False, 'showticklabels' : False, 'fixedrange' : True},
                        yaxis={'showgrid' : False, 'zeroline' : False, 'showticklabels' : False, 'fixedrange' : True},
                        annotations = [dict(
                            ax = arrows[i][0],
                            ay = arrows[i][1], axref='x', ayref='y',
                            x = arrows[i][2],
                            y = arrows[i][3], xref='x', yref='y',
                            showarrow=True,
                            arrowcolor='rgb(42, 63, 95)',
                            arrowhead=2,
                            arrowsize=3,
                            arrowwidth=1,
                            opacity=1
                    ) for i, e in enumerate(graph.edges)],
                    margin = {'l' : 0, 'r' : 0, 't' : 0, 'b' : 0}
                    ))
    html_repr = fig.to_html(full_html=False, config={'displayModeBar' : False})
    nodes_positions = {f'#module_{node}' : [int(graph.nodes[node]['position']['x']),
                                            int(graph.nodes[node]['position']['y'])]
                                            for node in graph.nodes}
    plot_id = re.search('<div id="([^"]*)"', html_repr).groups()[0]
    cback = """
    <script>
    var nodes_positions = {nodes_positions}
    var p_e = document.getElementById("{plot_id}");
    p_e.on('plotly_click', function(data){{
    var point = data.points[0];

    if (point){{
    let x = Math.trunc(point.x)
    let y = Math.trunc(point.y)
    for (let key in nodes_positions){{
    if (nodes_positions[key][0] === x) {{
    if (nodes_positions[key][1] === y){{
    window.open(key,'_self')
    }}
    }}
    }}
    }}
    }})
    </script>
    """.format(nodes_positions=nodes_positions,plot_id=plot_id)
    html_with_callback = """
    {html_repr}
    {cback}
    """.format(html_repr=html_repr, cback=cback)
    return html_with_callback


def super_thresholds_hint(module_id):
    df = element_metrics_dfs[str(module_id)].copy().set_index('element_id')
    messages = {}
    for idx, task in df.iterrows():
        hard_messages = []
        for metric in task.index:
            top_percent = None
            for i in range(4, -1, -1):
                if (np.isnan(df.loc[idx, metric])):
                    break
                if (metric != 'Доля решивших') & (df.loc[idx, metric] < super_thresholds[metric][i]):
                    break
                if (metric == 'Доля решивших') & (df.loc[idx, metric] > super_thresholds[metric][i]):
                    break    
                top_percent = i+1
            if top_percent != None:
                hard_messages.append([top_percent, metric])
        if hard_messages:
            messages[task.name] = hard_messages
    return messages


def create_personal_recomendations(module_id):
    messages = super_thresholds_hint(module_id)
    
    div = "<h4>Интересные факты о задачах в модуле:</h4>"
    if not messages:
        div += '<p>В текущем модуле нет задач, имеющих критические значения метрик</p>'
        return div
    div += '<ol>'
    
    for task in messages.keys():
        div += f'<li> Задача <b>{task}</b>:'
        div += '<ul>'
        for message in messages[task]:
            div += f'<li>Топ <b>{message[0]}%</b> сложных задач по метрике <b>"{message[1]}"</b>.</li>'
        div += '</ul>'
        div += '</li>'
    div += '</ol>'
    return div


def create_fans_downloader():
    fans_folders={re.findall('\(ID=[0-9]+\)', folder)[0][4:-1] : folder for folder in os.listdir(FAN_PATH)}
    def read_file(file):
        with open(file, 'r', encoding='UTF-8') as f:
            data = f.read()
        data = data.split('\n')
        return data
    fans = {}
    for id in course_module.id:
        task_fan = {}
        fan_folder = os.path.join(FAN_PATH, fans_folders[str(id)])
        task_fan = {re.sub('[0-9]+\_', '', file)[:-4]: read_file(os.path.join(fan_folder, file)) for file in os.listdir(fan_folder) if file.endswith('.csv')}
        fans.update(task_fan)
    datalist='''<datalist id='list'>
    '''
    for id in fans.keys():
        datalist+='''   <option value='{id}'>{id}</option>
        '''.format(id=id)
    datalist+='''
    </datalist>
    '''
    data ='''var data = {'''
    for id in fans.keys():
        data+=''''{id}' : {fan},
        '''.format(id=id, fan=fans[id])
    data+='''}'''
    html ='''
    <div id='download-fan' style='flex'>
        <input id='input' list='list' placeholder='Введите id задачи'/>
        {datalist}
        <button id='button'>
            Получить веера
        </button>
        <script>
        var button = document.getElementById('button')
        {data}
    ''' .format(datalist=datalist,data=data)
    html += '''
    function saveDoc(){
    var selector = document.getElementById('input')
    var val = selector.value
    saveData(val)
    }
    function saveData(val){
    var csv_data = data[val];
    if (csv_data === undefined){
        alert('К сожалению, задачи с id = '.concat(val).concat(' не найдено'))
    }
    let csv = "data:text/csv;charset=utf-8,";
    csv_data.forEach(function(dataArray){
    let row = dataArray;
    csv += row + "\\r\\n"
    });
    var encoded = encodeURI(csv);
    var link = document.createElement("a");
    link.setAttribute("href", encoded);
    link.setAttribute("download", val.concat(".csv"));
    document.body.appendChild(link);
    link.click()
    link.remove()
    }
    button.addEventListener('click', function(){saveDoc()})
    </script>
    </div>
    '''
    return html


def create_course_table():
    module_ids = [f'{course_module[course_module.id == id].title.iloc[0]} <br> (ID = {id})' 
                     for id in course_module.id]
    header = '<th style = "border: 1px solid black;"> </th> \n'
    for id in module_ids:
        header += f'<th style = "background-color : lightgrey; aligh : center; border: 1px solid black;"> {id} </th> \n'
    rows = [header]
    metrics = module_metrics_df.columns[1:]
    for metric in metrics:
        row = f'<td style="border : 1px solid black;">{metric}</td>'
        row = '''
        <td style="background-color: lightgrey; border: 1px solid black">
            <span class="metric-description">
                {metric}
                <span class="metric-description-text">
                    <p class="left nmargin">
                        <b class="underlined"> {metric}: </b>
                        <br>
                        {metric_description}
                    </p>
                </span>
            </span>
        </td>
        '''.format(metric=metric, metric_description=module_metrics_descriptions[metric])
  
        for id in course_module.id:
            value = module_metrics_df[module_metrics_df.module_id == id].loc[:, metric].round(2).iloc[0]
            color = colors(value, module_metrics_thresholds[metric])
            row += f'<td style="background-color : {color}; border : 1px solid black;"> {value} </td> \n'
        rows.append(row)
    return table_template.render(rows=rows)


def create_course_recomendations():
    r0 = ', '.join([str(r) for r in course_recomendations[0]])
    r1 = ', '.join([str(r) for r in course_recomendations[1]])
    r2 = ', '.join([str(r) for r in course_recomendations[2]])
    div = '''
    <div class="course-recomendations">
        <ul>
            <li>Задачи <b> {r0} </b> требуют у учеников много времени и попыток.</li>
            <li>Задачи <b> {r1} </b> часто пропускают, однако среди приступивших процент решений очень высок.</li>
            <li>О сложности задач <b> {r2}  </b> сигнализирует больше половины метрик.</li>
        </ul>
    </div>
    '''.format(r0=r0, r1=r1, r2=r2)
    return div


def create_course_minimap(module_id):
    graph = networkx.Graph().to_directed_class()
    graph = networkx.from_pandas_edgelist(course_graph, source='from_module_id', target='to_module_id', create_using=graph)
    positions = json.loads(course_graph_layout.content.iloc[0])['module_positions'] 
    for node in graph.nodes:
        x, y = positions[str(node)]['x'], -positions[str(node)]['y']
        graph.nodes[node]['position'] = {'x' : x, 'y' : y}
        graph.nodes[node]['id'] = node
        graph.nodes[node]['type'] = course_module[course_module.id == node].type.iloc[0]
    sources = set(course_graph.from_module_id.unique())
    destinitions = set(course_graph.to_module_id.unique())
    from_zero_node = list(sources.difference(destinitions))
    graph.add_node(0)
    graph.nodes[0]['position'] = {'x' : 0, 'y' : 0}
    graph.nodes[0]['id'] = ''
    graph.nodes[0]['type'] = 'graph_source'
    for node in from_zero_node:
        graph.add_edge(0, node)

    def create_graph_lines(graph):
        lines = []
        for e in graph.edges:
            x_from, y_from = graph.nodes[e[0]]['position']['x'], graph.nodes[e[0]]['position']['y']
            x_to, y_to = graph.nodes[e[1]]['position']['x'],  graph.nodes[e[1]]['position']['y']
            line = go.Scatter(x=(x_from, x_to, None), y=(y_from, y_to, None), hoverinfo='none',
                            #mode='lines',
                            marker=dict(symbol="arrow-bar-up", color='rgb(42, 63, 95)', angleref="previous", size = 1),
                            line_shape='spline')
            lines.append(line)
        return lines

    def create_graph_nodes(graph, module_id):
        nodes = go.Scatter(x=[], y=[], mode='markers', hoverinfo='none', customdata=[],
                           marker={'size' : 15,  'symbol' : [], 'color' : [], 'opacity' : 1, 'line' : {'width' : 1, 'color' : 'black'}})
        for node in graph.nodes:
            x = graph.nodes[node]['position']['x']
            y = graph.nodes[node]['position']['y']
            nodes['x'] += x,
            nodes['y'] += y,
            # text = graph.nodes[node]['id']
            # nodes['text'] += f'{text}',
            nodes['marker']['symbol'] += 'pentagon' if graph.nodes[node]['type'] == 'advanced' else 'circle', 
            if graph.nodes[node]['type'] == 'graph_source':
                color = 'lightgrey'
            elif node == module_id:
                color = 'lightgreen'
            else:
                color = 'white'
            nodes['marker']['color'] += color,
        return nodes


    def calculate_arrow_pos(e0, e1):
        if e1['type'] == 'ordinary': 
            r = 35
        else:
            r = 30
        
        x0, y0 = e0['position']['x'], e0['position']['y']
        x1, y1 = e1['position']['x'], e1['position']['y']
        l = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        vx = (x1 - x0) * (1 - r/l)
        vy = (y1 - y0) * (1 - r/l)
        x = x0 + vx
        y = y0 + vy
        ax = x1 - vx
        ay = y1 - vy 
        return ax, ay, x, y

    arrows = [calculate_arrow_pos(graph.nodes[e[0]], graph.nodes[e[1]]) for e in graph.edges]
    fig = go.Figure()
    lines = create_graph_lines(graph)

    for line in lines:
        fig.add_trace(line)
    fig.add_trace(create_graph_nodes(graph, module_id))

    fig.update_layout(dict(height=200, width=200,
                        plot_bgcolor='white',
                        showlegend=False,
                        hovermode='closest',
                        xaxis={'showgrid' : False, 'zeroline' : False, 'showticklabels' : False, 'fixedrange' : True},
                        yaxis={'showgrid' : False, 'zeroline' : False, 'showticklabels' : False, 'fixedrange' :True},
                        annotations = [dict(
                            ax = arrows[i][0],
                            ay = arrows[i][1], axref='x', ayref='y',
                            x = arrows[i][2],
                            y = arrows[i][3], xref='x', yref='y',
                            showarrow=True,
                            arrowcolor='rgb(42, 63, 95)',
                            arrowhead=2,
                            arrowsize=2,
                            arrowwidth=1,
                            opacity=1
                    ) for i, e in enumerate(graph.edges)],
                    margin = {'l' : 0, 'r' : 0, 't' : 0, 'b' : 0}
                    ))
    html_repr = fig.to_html(full_html=False, config={'displayModeBar' : False}, include_plotlyjs=False)
    nodes_positions = {f'#module_{node}' : [int(graph.nodes[node]['position']['x']),
                                            int(graph.nodes[node]['position']['y'])]
                                            for node in graph.nodes}
    plot_id = re.search('<div id="([^"]*)"', html_repr).groups()[0]
    cback = """
    <script>
    var nodes_positions = {nodes_positions}
    var p_e = document.getElementById("{plot_id}");
    p_e.on('plotly_click', function(data){{
    var point = data.points[0];

    if (point){{
    let x = Math.trunc(point.x)
    let y = Math.trunc(point.y)
    for (let key in nodes_positions){{
    if (nodes_positions[key][0] === x) {{
    if (nodes_positions[key][1] === y){{
    window.open(key,'_self')
    }}
    }}
    }}
    }}
    }})
    </script>
    """.format(nodes_positions=nodes_positions,plot_id=plot_id)
    html_with_callback = """
    {html_repr}
    {cback}
    """.format(html_repr=html_repr, cback=cback)
    return html_with_callback


def create_module_factoids(module_id):
    metrics = module_metrics_df[module_metrics_df.module_id == module_id]
    metric_names = module_metrics_df.columns[1:]
    factoids = []
    for name in metric_names:
        value = metrics[name].round(2).iloc[0]
        title = name + ', мин.' if name == 'Среднее время' else name
        factoids.append(factoid_temp.render(value=value, title=title))
    return factoids


def get_cell_style(t,b,l,r, bg):
    border = f"background-color : {bg};"
    border += f"border-top : 1px solid {t};"
    border += f"border-bottom : 1px solid {b};"
    border += f"border-left : 1px solid {l};"
    border += f"border-right: 1px solid {r};"
    return border


def create_module_table(module_id):
    current_module_ce = course_element[course_element.module_id == module_id]
    merged_with_metrics = pd.merge(current_module_ce, element_metrics_dfs[f'{module_id}'], how='outer', on='element_id').round(2).sort_values('position')
    merged_with_metrics = merged_with_metrics.set_index(merged_with_metrics.position).fillna('')
    ids = list(merged_with_metrics.element_id)
    types = list(merged_with_metrics.element_type)
    metric_names = list(merged_with_metrics.loc[:,'position':].columns[1:])
    header_borders = ('black',) * 4
    style = get_cell_style(*(header_borders + ('white',)))
    header = f'<th style = "{style}"></th>'
    style = get_cell_style(*(header_borders + ('lightgrey',)))
    for id in ids:
        header += f'<th style="{style}">{id}</th>'
    rows = [header]
    pic_row = len(metric_names) // 2
    for i, metric in enumerate(metric_names):
        row = '''
        <td style="{style}">
            <span class="metric-description">
                {metric}
                <span class="metric-description-text">
                    <p class="left nmargin">
                        <b class="underlined"> {metric}: </b>
                        <br>
                        {metric_description}
                    </p>
                </span>
            </span>
        </td>
        '''.format(style=style, metric=metric, metric_description=metrics_descriptions[metric])
        data = list(merged_with_metrics.loc[:, metric])
        for j, value in enumerate(data):
            if types[j] != 'task':
                cell = get_cell_style('#CDC2E1','#CDC2E1','black', 'black','#CDC2E1')
                if i == len(metric_names) - 1:
                    cell = get_cell_style('#CDC2E1','black','black', 'black','#CDC2E1')
                if types[j] == 'text' and i == pic_row:
                    value = '&#127344'
                if types[j] == 'video' and i == pic_row:
                    value = '&#9199'
                row += f'<td style="{cell}">{value}</td>'
            if types[j] == 'task':
                color = colors(value, metrics_thresholds[metric])
                cell = ('black',) * 4 
                cell += (color,)
                cell = get_cell_style(*cell)
                row += f'<td style="{cell}">{value}</td>'
        rows.append(row)
    return table_template.render(table_class='module-table', rows=rows)


def create_module_bar(module_id):
    current_module_ce = course_element[course_element.module_id == module_id]
    merged_with_metrics = pd.merge(current_module_ce, element_metrics_dfs[f'{module_id}'], how='outer', on='element_id').sort_values('position')
    merged_with_metrics = merged_with_metrics.set_index(merged_with_metrics.position)
    merged_with_metrics = merged_with_metrics.transpose()
    element_types = np.array(merged_with_metrics.loc['element_type', :])
    element_ids = np.array(merged_with_metrics.loc['element_id', :])
    merged_with_metrics = merged_with_metrics.drop(['id', 'module_id', 'element_type','is_advanced', 'max_tries', 'score', 'position', 'element_id']).astype(float)
    h=350
    def module_bar(metric, merged_with_metrics):
        fill_value = np.max(merged_with_metrics.loc[metric, '1':])
        if fill_value == 0:
            fill_value = 1
        not_task_positions = []
        task_positions = []
        not_task_text = []
        for i in merged_with_metrics.columns:
            if element_types[i - 1] != 'task':
                not_task_positions.append(i)
                if element_types[i-1] == 'text':
                    not_task_text.append('\U0001F170')
                else:
                    not_task_text.append('\u23ef')
            else:
                task_positions.append(i)
        y1 = merged_with_metrics.loc[metric, task_positions]
        
        not_tasks_bar = go.Bar(hoverinfo='none', y = [fill_value] * len(not_task_positions), x = not_task_positions, marker_color ='#CDC2E1', 
                          text = not_task_text, marker_line={'width':1, 'color':'black'},showlegend=False)
        bar_color = [colors(y, metrics_thresholds[metric]) for y in y1]
        tasks_bar = go.Bar(hoverinfo='none', name=metric, y = y1, x=task_positions, marker_color = bar_color, showlegend=False, marker_line={'width':1, 'color':'black'})
        return tasks_bar, not_tasks_bar
    fig = go.Figure()
    buttons = []
    for i, metric in enumerate(metrics_thresholds.keys()):
        tasks, not_tasks = module_bar(metric, merged_with_metrics)
        fig.add_trace(tasks)
        fig.add_trace(not_tasks)
        if metric == 'Доля сделавших много попыток':
            metric = 'Доля сделавших <br> много попыток'
        buttons.append({'label' : metric, 'method' : 'update', 'args' : 
                        [{'visible' : ([False] * 2)* i + ([True] * 2) + ([False] * 2) * (len(metrics_thresholds.keys()) - i)}]})
    for i in range(2, 2 * len(metrics_thresholds.keys())):
        fig.update_traces(visible=False, selector=i)

    fig.update_layout(height = h, margin = {'l' : 102, 'r' : 0, 't' : 50, 'b' : 50})
    fig.update_layout(xaxis = dict(tickvals=list(range(1, merged_with_metrics.shape[1]+1)),ticktext=list(element_ids), tickfont=dict(size=12),
                                   fixedrange=True),
                    xaxis_tickangle=-45, 
                    yaxis = dict(fixedrange=True))
    fig.update_layout(updatemenus = [{
        'active' : 0,
        'showactive' : True,
        'buttons' : buttons,
        'pad' : {'r' : 10, 't' : 0},
        'x' : 0,
        'xanchor' : 'left',
        'y' : 1.2,
        'yanchor' : 'top',
        'font' : {'size' : 12}

        # 'x' : 0.01,
        # 'y' : 1,
    }])
    return fig.to_html(full_html=False, include_plotlyjs = False, config={'displayModeBar' : False})


def create_module_report(module_id):
    module_name = f'{course_module[course_module.id == module_id].title.iloc[0]} (ID = {module_id})'
    minimap = create_course_minimap(module_id)
    factoids = create_module_factoids(module_id)
    recomendations = create_personal_recomendations(module_id)
    table = create_module_table(module_id)
    bar = create_module_bar(module_id)
    module = module_template.render(id=f'module_{module_id}',module_name=module_name, minimap=minimap,
                                    factoids=factoids, personal_recomendations=recomendations,
                                    module_table=table, module_bar=bar)
    return module
########################################################
# Загрузка готовых таблиц с данными
########################################################
# Таблицы по курсу:
course = pd.read_csv(os.path.join(COURSE_INFO_FOLDER,'course.csv'))
course_graph = pd.read_csv(os.path.join(COURSE_INFO_FOLDER, 'course_graph.csv'))
course_graph = course_graph[course_graph.course_id == COURSE_ID]
course_graph_layout = pd.read_csv(os.path.join(COURSE_INFO_FOLDER, 'course_graph_layout.csv'))
course_graph_layout = course_graph_layout[course_graph_layout.course_id == COURSE_ID]
course_module = pd.read_csv(os.path.join(COURSE_INFO_FOLDER, 'course_module.csv'))
course_module = course_module[course_module.course_id == COURSE_ID]
course_element = pd.read_csv(os.path.join(COURSE_INFO_FOLDER, 'course_element.csv'))
user_course_progress = pd.read_csv(os.path.join(COURSE_INFO_FOLDER, 'user_course_progress.csv'))
user_course_progress = user_course_progress[user_course_progress.course_id == COURSE_ID]
user_module_progress = pd.read_csv(os.path.join(COURSE_INFO_FOLDER, 'user_module_progress.csv'))
user_module_progress = user_module_progress[user_module_progress.course_id == COURSE_ID]
user_element_progress = pd.read_csv(os.path.join(COURSE_INFO_FOLDER, 'user_element_progress.csv'))
user_element_progress = user_element_progress[user_element_progress.course_id == COURSE_ID]
# Таблицы с метриками:
module_metrics_df = pd.read_csv(os.path.join(METRICS_FOLDER, 'module_metrics.csv'))
element_metrics_dfs = {file[:-4] : pd.read_csv(os.path.join(METRICS_FOLDER, file)) 
                       for file in os.listdir(METRICS_FOLDER) if file.endswith('.csv')
                       and file != 'module_metrics.csv'}
# Трешходы и описание метрик:
with open(os.path.join(METRICS_FOLDER, 'element_metrics_info.json'), 'r') as f:
    element_metrics_info = json.load(f)
metrics_thresholds = {i : element_metrics_info[i]['threshold'] for i in element_metrics_info.keys()}
metrics_descriptions = {i : element_metrics_info[i]['description'] for i in element_metrics_info.keys()}
with open(os.path.join(METRICS_FOLDER, 'module_metrics_info.json'), 'r') as f:
    module_metrics_info = json.load(f)
module_metrics_thresholds = {i : module_metrics_info[i]['threshold'] for i in module_metrics_info.keys()}
module_metrics_descriptions = {i : module_metrics_info[i]['description'] for i in module_metrics_info.keys()}
with open(os.path.join(METRICS_FOLDER, 'super_thresholds.json')) as f:
    super_thresholds = json.load(f)
# Рекомендации по курсу:
with open(os.path.join(METRICS_FOLDER, 'course_recomendations.json'), 'r') as f:
    course_recomendations = json.load(f)
########################################################
# Загрузка шаблонов:
########################################################
with open(CSS_FILE, 'r') as f:
    css = f.read()
loader = FileSystemLoader(searchpath=TEMPLATE_PATH)
env = Environment(loader=loader)
body_temp = env.get_template('body.html')
header_temp = env.get_template('header.html')
factoid_temp = env.get_template('factoid.html')
map_template = env.get_template('map.html')
table_template = env.get_template('table.html')
module_template = env.get_template('module.html')
########################################################
# Заполнение шаблонов и сохранение HTML
########################################################

start, close = extract_dates()
course_factoids = create_course_factoids()
header = header_temp.render(course_id=COURSE_ID,
                            start=start,
                            close=close,
                            course_factoids=course_factoids
)
course_recomendations = create_course_recomendations()
fans_downloader = create_fans_downloader()
graph = create_course_graph()
map = map_template.render(course_graph=graph)
course_table = create_course_table()
modules_list =[create_module_report(id) for id in course_module.id]
body = body_temp.render(style_sheet=css, header=header, map=map,
                        course_table = course_table, course_recomendations=course_recomendations, 
                        fans_downloader=fans_downloader,
                        modules_list = modules_list)
with open(OUTPUT_FILE, 'w', encoding="utf-8") as f:
    f.write(body)
