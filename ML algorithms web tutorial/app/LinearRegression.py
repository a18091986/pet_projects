import pandas as pd
import numpy as np

from dash import Dash, Input, Output, State, callback, dash_table, html, dcc, ctx
import dash_bootstrap_components as dbc
from DataFrame_primary_preprocessing import df as initial_df, columns_names as columns_names
import plotly.express as px

from itertools import combinations

from dash.exceptions import PreventUpdate

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)

app.layout = dbc.Container([
    dcc.Store(id='prevent_error_if_show_nan_before_fill'),
    dcc.Store(id='df_after_fill_na_0'),
    dcc.Store(id='df_after_fill_na_1'),
    dcc.Store(id='for_graph_region'),

    html.H3('Демонстрация работы модели линейной регрессии', style={'textAlign': 'center', 'margin': '20px'}),
    html.H4('ИСХОДНЫЙ ДАТАСЕТ', style={'textAlign': 'center', 'margin': '20px', 'color': '#D35400'}),
    html.H5('Выберите столбцы (используйьте иконку в шапке столбца для удаления)', style={'textAlign': 'center', 'margin': '20px', 'color': '#D35400'}),
    html.Div([dash_table.DataTable(
        data=initial_df.to_dict('records'),
        columns=[{"name": i, "id": i, 'deletable': True, 'renamable': False} for i in initial_df.columns],
        id='main_table',
        page_size=7,  # пагинация
        style_table={'overflowX': 'scroll'},
        style_cell_conditional=[
            {
                'textAlign': 'center'
            }

        ],
        style_cell={'padding': '5px'},
        style_header={
            'backgroundColor': '#FFFF73',
            'fontWeight': 'bold',
        },
        style_data={
            'color': 'blue',
            'backgroundColor': 'white'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#C8C8C8',
            },
        ]
    ), ], id='Main_Table'),
    html.Hr(),
    dbc.Row([
        dbc.Col([dbc.Button("Вернуть исходное состояние таблицы", color="primary",
                            id='reset_main_table_btn')],
                class_name='d-grid gap-2', width={"size": 6, "offset": 0}),
        dbc.Col([dbc.Button("Применить изменения и показать данные по пропускам", color="primary",
                            id='submit_changes_to_main_table_btn')],
                class_name='d-grid gap-2', width={"size": 6, "offset": 0}),
    ], style={'margin': '5px'}),
    html.Hr(),
    html.Div(id='nan_info_with_buttons'),
    html.Div(id='table_after_fill_na'),
    html.Div(id='graph_region_control', className="d-grid gap-2", style={'margin-bottom': '20px'}),
    html.Div(id='graph_region'),
])


@callback(
    Output('main_table', 'data'),
    Output('main_table', 'columns'),
    Input('reset_main_table_btn', 'n_clicks'),
)
def reset_changes_on_main_table(n):
    '''
    возвращает главную таблицу в исходный вид
    '''
    df_cur = initial_df.copy()
    return df_cur.to_dict('records'), \
           [{"name": i, "id": i, 'deletable': True, 'renamable': False} for i in df_cur.columns],


@callback(
    Output('nan_info_with_buttons', 'children'),
    Output('prevent_error_if_show_nan_before_fill', 'data'),
    Input('submit_changes_to_main_table_btn', 'n_clicks'),
    Input('df_after_fill_na_1', 'data'),
    State('main_table', 'data'),
    prevent_initial_call=True
)
def show_nans_after_main_table(n, data_from_dccStore, data_from_main_table):
    '''
    отображает пропущенные значения исходной таблицы и кнопки заполнения пропусков
    '''
    out_row_list = []
    if ctx.triggered_id == 'submit_changes_to_main_table_btn':
        df_cur = pd.DataFrame(data_from_main_table)
    else:
        df_cur = pd.read_json(data_from_dccStore)
    out_row_list.append(
        html.H4('ИНФОРМАЦИЯ О ПРОПУСКАХ', style={'textAlign': 'center', 'margin': '20px', 'color': '#D35400'}))
    out_row_list.append(dbc.Row([
        dbc.Col(['Наименование признака'], width=3, style={'textAlign': 'center'}),
        dbc.Col(['Количество пропусков'], width=3, style={'textAlign': 'center'}),
        dbc.Col(['Метод заполнения пропусков в каждой колонке'], width=4, style={'textAlign': 'center'}),
    ], justify='center', style={'margin-top': '3px'}))
    out_row_list.append(html.Hr())
    for column in df_cur.columns.to_list():
        out_row_list.append(dbc.Row([
            dbc.Col([column], width=3, style={'textAlign': 'center'}),
            dbc.Col([df_cur[column].shape[0] - df_cur.count().to_frame().T.to_dict('records')[0][column]],
                    width=3, style={'textAlign': 'center'}),
            dbc.Col([
                dbc.InputGroup([
                    dbc.Select(
                        options=[
                            {"label": "Среднее", "value": 'mean'},
                            {"label": "Медиана", "value": 'median'},
                            {"label": "Мода", "value": 'mode'},
                        ],
                        value='mean',
                        size='sm', ),

                ], size='sm'),
            ], style={'textAlign': 'center'}, width=4),
        ], justify='center', style={'margin-top': '3px'}))
    out_row_list.append(html.Hr())
    out_row_list.append(
        dbc.Row([
            dbc.Col([dbc.Button("Заполнить пропуски в соответствии с выбранными методами",
                                color="primary", id='fillna_btn')],
                    class_name='d-grid gap-2', width={"size": 6, "offset": 0}),
            dbc.Col([dbc.Button("Отобразить текущую информацию по пропускам", color="primary",
                                id='show_nan_info_btn')],
                    class_name='d-grid gap-2', width={"size": 6, "offset": 0}),
        ]))
    out_row_list.append(html.Hr())
    return out_row_list, df_cur.to_json()


@callback(
    Output('table_after_fill_na', 'children'),
    Output('df_after_fill_na_0', 'data'),
    Output('for_graph_region', 'data'),
    Output('graph_region_control', 'children'),
    Input('fillna_btn', 'n_clicks'),
    State('nan_info_with_buttons', 'children'),
    State('main_table', 'data'),
    prevent_initial_call=True
)
def generate_df_after_fillna_click_btn(n, nan_info_with_buttons_div_structure,
                                       current_data_in_main_table):
    '''Выдает df c заполненными пропусками'''
    if n is None:
        raise PreventUpdate
    df_cur = pd.DataFrame(current_data_in_main_table)
    for item in nan_info_with_buttons_div_structure:
        try:
            column = item['props']['children'][0]['props']['children'][0]
            method = item['props']['children'][2]['props']['children'][0]['props']['children'][0]['props']['value']
            if method == 'mean':
                df_cur[column].fillna(round(df_cur[column].mean(), 5), inplace=True)
            elif method == 'mode':
                df_cur[column].fillna(round(df_cur[column].mode()[0], 5), inplace=True)
            else:
                df_cur[column].fillna(round(df_cur[column].median(), 5), inplace=True)
        except Exception:
            pass
    out_list = [
        html.H4('ДАТАСЕТ ПОСЛЕ ЗАПОЛНЕНИЯ ПРОПУСКОВ',
                style={'textAlign': 'center', 'margin': '20px', 'color': '#D35400'}),
        dash_table.DataTable(
            data=df_cur.to_dict('records'),
            columns=[{"name": i, 'id': i} for i in df_cur.columns],
            id='table_after_fillna',
            page_size=7,  # пагинация
            style_table={'overflowX': 'scroll'},
            style_cell_conditional=[
                {
                    'textAlign': 'center'
                }
            ],
            style_cell={'padding': '5px'},
            style_header={
                'backgroundColor': '#FFFF73',
                'fontWeight': 'bold',
            },
            style_data={
                'color': 'blue',
                'backgroundColor': 'white'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#C8C8C8',
                },
            ]
        ), html.Hr()]
    dropdown_menu_items = list(combinations(df_cur.columns.to_list(), 2))
    menu_items_list_options = []
    for item in dropdown_menu_items:
        menu_items_list_options.append({'label': f'{item[0]} vs {item[1]}', 'value': f'{item[0]} vs {item[1]}'})
    out_list_for_graph_region_control = [
        html.H4('ОТОБРАЖЕНИЕ ГРАФИКОВ ПОПАРНОЙ ЗАВИСИМОСТИ ПРИЗНАКОВ',
                style={'textAlign': 'center', 'margin': '20px', 'color': '#D35400'}),
        dbc.InputGroup([
            dbc.Select(
                options=menu_items_list_options,
                value=np.random.choice(menu_items_list_options)['value'],
                size='sm', id='x_y'),
        ], size='sm'),
    ]
    return out_list, df_cur.to_json(), df_cur.to_json(), out_list_for_graph_region_control


@callback(
    Output('df_after_fill_na_1', 'data'),
    Input('show_nan_info_btn', 'n_clicks'),
    State('df_after_fill_na_0', 'data'),
    State('prevent_error_if_show_nan_before_fill', 'data'),
    prevent_initial_call=True
)
def show_nans_after_main_table_after_fillna(n, data, data_):
    '''
    отображает пропущенные значения исходной таблицы и кнопки заполнения пропусков
    '''
    if not data:
        return data_
    return data


@callback(
    Output('graph_region', 'children'),
    Input('x_y', 'value'),
    State('for_graph_region', 'data'),
    prevent_initial_call=True
)
def show_graph(data_x_y, data):
    '''
    отрисовывает график зависимости одной переменной от другой
    '''
    x, y = data_x_y.split(' vs ')
    df_cur = pd.read_json(data)
    fig = px.scatter(df_cur, x = x, y = y)


    return [dcc.Graph(id='graph', figure=fig)]


if __name__ == "__main__":
    app.run_server(debug=True, host='192.168.2.23', port=8050)
