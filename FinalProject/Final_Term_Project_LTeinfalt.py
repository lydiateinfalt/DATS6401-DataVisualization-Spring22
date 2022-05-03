#Lydia Teinfalt
# Mushroom Final Term Project
#04/06/2022
import dash
import matplotlib.pyplot as plt
from dash import html
from dash import dcc
import numpy as np
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import math as m
from dash import callback_context

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#Main
my_app = dash.Dash("My app", external_stylesheets= external_stylesheets)

#Question 1 Data -----------------------------------------------------------------------------------------

df= pd.read_csv("mushrooms.csv")
df = df.dropna(axis=0, how='any')
columns = df.columns

#-------------------------------------------------------------------------------------------------
# Create a Dash layout
my_app.layout = html.Div([
    html.H1("Mushrooms Final Term Project", style={'textAlign':'center'}),
    html.Br(),
    dcc.Tabs(id = 'hw-questions',
             children=[
                dcc.Tab(label = 'Pie Charts', value = 'q1'),
                dcc.Tab(label = 'Classification', value = 'q2'),
                dcc.Tab(label = 'Violin/Scatter Plots', value='q3'),
                dcc.Tab(label='Box Plots', value='q4'),
                dcc.Tab(label='Dashboard', value='q6'),
             ]),
    html.Div(id = 'layout')
])

tab1_layout = html.Div([
    html.H4('Mushroom Pie Chart'),
    html.P('Please select the feature from the menu'),
    dcc.Dropdown(
        id='my-drop',
        options=[
            {'label': 'gill color', 'value': 'gill-color'},
            {'label': 'cap color', 'value': 'cap-color'},
            {'label': 'stem color', 'value': 'stem-color'},
        ], clearable=False
    ),
    html.Br(),
    dcc.Dropdown(
        id='my-drop2',
        options=[
            {'label': 'cap diameter', 'value': 'cap-diameter-norm'},
            {'label': 'stem height', 'value': 'stem-height-norm'},
            {'label': 'stem width', 'value': 'stem-width-norm'},
        ], clearable=False
    ),
    html.Br(),
    dcc.Graph(id = 'my-graph'),
])

tab2_layout = html.Div([
    html.H4('Mushroom Classification'),
    html.P('What are common stem colors?'),
    dcc.RadioItems(
        options=[
            {'label': 'White', 'value': 'white'},
            {'label': 'Brown', 'value': 'brown'},
            {'label': 'Yellow', 'value': 'yellow'},
            {'label': 'Gray', 'value': 'gray'},
            {'label': 'Orange', 'value': 'orange'},
            {'label': 'Red', 'value': 'red'},
            {'label': 'Purple', 'value': 'purple'},
            {'label': 'Pink', 'value': 'pink'},
            {'label': 'Black', 'value': 'black'},
            {'label': 'Green', 'value': 'green'},
            {'label': 'Buff', 'value': 'buff'},
        ],
        value='white',
    ),
    html.Br(),
    html.P('What are common cap shapes?'),
    dcc.RadioItems(
        options=[
            {'label': 'Flat', 'value': 'flat'},
            {'label': 'Convex', 'value': 'convex'},
            {'label': 'Bell', 'value': 'bell'},
            {'label': 'Conical', 'value': 'conical'},
            {'label': 'Sunken', 'value': 'sunken'},
            {'label': 'Spherical', 'value': 'spherical'},
            {'label': 'Other', 'value': 'other'},
        ],
        value='bell',
    ),
    html.Br(),
    html.P('What is the approximate stem height (cm)?'),
    dcc.Slider(id = "input2",min = 0, max = 33.9, value = 10, marks = {-0:'0',5: '5', 10: '10', 20:'20', 30:'30'}),
    html.Br(),
    html.P('What is the approximate stem width (cm)?'),
    dcc.Slider(id = "input3",min = -2.36, max = 2, value = 1, marks = {-2:'-2',-1: '-1', 0: '0', 1:'1', 2:'2'}),
    html.Br(),
    html.P('What is approximate cap diameter (cm)?'),
    dcc.Slider(id='n', min=-2.36, max=4, value=1, marks={-2: '-2', -1: '-1', 0: '0', 1: '1', 2: '2', 3: '3',4: '4'}),
    html.Br(),
    html.P('What is your name?'),
    dcc.Input(id = 'input1', type = 'text'),
    html.Div(id = 'output2'),
])

# fig31 = px.violin(df, x = 'season', y = 'stem-width-norm')
tab3_layout = html.Div([
    html.H4('Violinplot or Scatter'),
    html.Button('Violinplot', id='button-1'),
    html.Button('Scatter', id='button-2'),
    html.Div(id='container-button-timestamp'),
    html.Br(),
    dcc.Graph(id='my-graph3'),
])

tab4_layout = html.Div([
    html.H4("Analysis of Mushroom Habitat and Season"),
    html.P("x-axis:"),
    dcc.Checklist(
        id='x-axis',
        options=[
            {'label': 'Season', 'value': 'season'}
        ],
        value='season',
        inline=True
    ),
    html.P("y-axis:"),
    dcc.RadioItems(
        id='y-axis',
        options=[
            {'label': 'Stem Width', 'value': 'stem-width-norm'},
            {'label': 'Stem Height', 'value': 'stem-height-norm'},
            {'label': 'Cap Diameter', 'value': 'cap-diameter'},
        ],
        value='stem-width-norm',
        inline=True,
    ),
    dcc.Graph(id = 'graph4'),
])

df_bar = df[['stem-color', 'cap-shape', 'gill-color', 'stem-height-norm', 'cap-diameter-norm','class']]
stc= pd.DataFrame(df['stem-color'].value_counts())
cpc= pd.DataFrame(df['cap-shape'].value_counts())
gcc = pd.DataFrame(df['gill-color'].value_counts())
ccc = pd.DataFrame(df['cap-color'].value_counts())
rtc = pd.DataFrame(df['ring-type'].value_counts())
hc = pd.DataFrame(df['habitat'].value_counts())
# sc = pd.DataFrame(df['season'].value_counts())

# fig61 = px.bar(stc, x=stc.index, y="stem-color", barmode="group")
fig61 = px.box(data_frame=df, x='habitat', y='stem-height-norm', color='class')
# fig62 = px.bar(cpc, x=cpc.index, y="cap-shape", barmode="group")
fig62 = px.pie(df, values='stem-width-norm', names='habitat')
fig65 = px.bar(gcc, x=gcc.index, y="gill-color", barmode="group")
fig64 = px.bar(ccc, x=ccc.index, y="cap-color", barmode="group")
fig63 = px.bar(stc, x='stem-color', y=stc.index, barmode="group", orientation='h')
fig66 = px.bar(rtc, x=rtc.index, y="ring-type", barmode="group")
tab6_layout = html.Div(children=[
    html.Div([
        html.Div([
            html.H4("Box Plot: Stem Height vs Habitat by Class"),
            dcc.Graph(
                id='graph61',
                figure=fig61
            ),
            # html.H5("Slider 1"),
            # dcc.Slider(id='n', min=0, max=20, value=1),
        ], className='six columns'),
        html.Div([
            html.H4("Bar Plot: Cap-Shape Counts"),

            dcc.Graph(
                id='graph62',
                figure=fig62
            ),
            # html.H5("Slider 2"),
            # dcc.Slider(id='n', min=0, max=20, value=1),
        ], className='six columns'),
    ], className='row'),
    # New Div for all elements in the new 'row' of the page
    html.Div([
        html.Div([
            html.H4("Bar Plot: Gill-Color Counts"),
            dcc.Graph(
                id='graph63',
                figure=fig63
            ),
            # html.H5("Slider 3"),
            # dcc.Slider(id='n', min=0, max=20, value=1),
        ], className='six columns'),
        html.Div([
            html.H4("Bar Plot: Cap-Color Counts"),
            dcc.Graph(
                id='graph64',
                figure=fig64
            ),
            # html.H5("Slider 4"),
            # dcc.Slider(id='n', min=0, max=20, value=1),
        ], className='six columns'),
    ], className='row'),
])

@my_app.callback(Output(component_id='layout', component_property='children'),
            [Input(component_id='hw-questions', component_property='value')])
def update_layout(ques):
    if ques == 'q1':
        return tab1_layout
    elif ques == 'q2':
        return tab2_layout
    elif ques == 'q3':
        return tab3_layout
    elif ques == 'q4':
        return tab4_layout
    else:
        return tab6_layout

@my_app.callback(
    Output(component_id='my-graph', component_property='figure'),
    [Input(component_id='my-drop', component_property='value'),
     Input(component_id='my-drop2', component_property='value')]
)
def display_graph(input1, input2):
    fig = px.pie(df, values=input2, names = input1, title="Mushroom Pie Plot")
    return fig

@my_app.callback(
    Output(component_id='output2', component_property='children'),
    [Input(component_id='input1', component_property='value')]
)

def update_q1(input):
    return f'{input}, your mushroom might be poisonous.'

# Source: https://dash.plotly.com/dash-html-components/button
@my_app.callback(
    [Output('container-button-timestamp', 'children'),
     Output(component_id='my-graph3', component_property='figure')],
    Input('button-1', 'n_clicks'),
    Input('button-2', 'n_clicks'),
)
def displayClick(btn1, btn2):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'button-1' in changed_id:
        msg = 'Violin plot was most recently clicked'
        fig = px.violin(data_frame=df, x = 'season', y = 'stem-height-norm', color='class')
    elif 'button-2' in changed_id:
        msg = 'Scatter plot was most recently clicked'
        fig = px.scatter(data_frame=df, x='stem-width-norm', y='stem-height-norm', color='class', trendline='ols')
    return html.Div(msg), fig

@my_app.callback(
    Output(component_id="graph4", component_property="figure"),
    [Input(component_id="x-axis", component_property="value"),
    Input(component_id="y-axis", component_property="value")])
def generate_chart(x, y):
    fig = px.box(df, x=x, y=y)
    return fig


@my_app.callback(
    [Output(component_id='graph61', component_property='figure'),
    Output(component_id='graph62', component_property='figure'),
    Output(component_id='graph63', component_property='figure'),
    Output(component_id='graph64', component_property='figure')]
)
def display_dashboard():
    return ""


my_app.run_server(port = 8080, host='0.0.0.0')
