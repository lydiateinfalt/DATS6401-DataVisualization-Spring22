import dash
from dash import html
from dash import dcc
import numpy as np
from dash.dependencies import Input, Output
import plotly.express as px
from scipy.fft import fft

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWlwgP.css']
my_app = dash.Dash("My app", external_stylesheets= external_stylesheets)

# Create a Dash layout
my_app.layout = html.Div([
    html.H1("HW 4", style={'textAlign':'center'}),
    html.Br(),
    dcc.Tabs(id = 'hw-questions',
             children=[
                dcc.Tab(label = 'Question 1', value = 'q1'),
                dcc.Tab(label = 'Question 2', value = 'q2'),
                dcc.Tab(label = 'Question 3', value='q3'),
             ]),
    html.Div(id = 'layout')
])

tab1_layout = html.Div([
    html.H1('Question1'),
    html.H3('Change the value in the textbox to see callsbacks in action'),
    html.P('Input:'),
    dcc.Input(id = 'input1', type = 'text' ),
    html.Br(),
    html.Div(id='output1')
])


tab2_layout = html.Div([
    html.H1('Question 2: Sinusoidal Wave '),
    html.P('Number of cycles of sinusoidal:'),
    dcc.Input(id='input1', type="number"),
    html.P('Mean of the white noise:'),
    dcc.Input(id='input2', type='number'),
    html.P('Standard deviation of the white noise'),
    dcc.Input(id='input3', type='number'),
    html.P("Number of samples:"),
    # dcc.Slider(id='n', min=1, max=1000, value=100, marks={10: '10', 50: '50', 100: '100', 500: '500', 1000: '1000'}),
    dcc.Input(id = 'n', max=1000, value=100),
    # html.Div(id='slider-output-container'),
    html.Br(),
    dcc.Graph(id="graph2"),
    html.Br(),
    dcc.Graph(id="graph3"),
])

tab3_layout = html.Div([
    html.H1('Complex Data Visualization'),
    dcc.Dropdown(id = 'drop3',
                 options = [
                    {'label': 'Introduction', 'value': 'Introduction'},
                    {'label': 'Panda Package', 'value': 'Panda Package'},
                    {'label': 'Seaborn Package', 'value': 'Seaborn Package'},
                    {'label': 'Matplotlib Package', 'value': 'Matplotlib Package'},
                    {'label': 'Principal Component Analysis', 'value': 'Principal Component Analysis'},
                    {'label': 'Outlier Detection', 'value': 'Outlier Detection'},
                    {'label': 'Interactive Visualization', 'value': 'Interactive Visualization'},
                    {'label': 'Web-based App using Dash', 'value': 'Web-based App using Dash'},
                    {'label': 'Tableau', 'value': 'Tableau'}
                 ], value = 'Introduction'),
    html.Br(),
    html.Div(id = 'output3')
])

@my_app.callback(Output(component_id='layout', component_property='children'),
            [Input(component_id='hw-questions', component_property='value')])
def update_layout(ques):
    if ques == 'q1':
        return tab1_layout
    elif ques == 'q2':
        return tab2_layout
    else:
        return tab3_layout

@my_app.callback(
    Output(component_id='output1', component_property='children'),
    [Input(component_id='input1', component_property='value')]
)

def update_q1(input):
    return f'The output value is {input}'

@my_app.callback(
    [Output(component_id='graph2', component_property='figure'),
    Output(component_id='graph3', component_property='figure')],
    [Input(component_id='input1', component_property='value'),
    Input(component_id='input2', component_property='value'),
    Input(component_id='input3', component_property='value'),
    Input(component_id='n', component_property='value')])
def display_graph(a,b,c,n):
    x = np.linspace(-np.pi, np.pi, num=n)
    noise = np.random.normal(loc=b, scale=c, size=n)
    xn= a*(x+ noise)
    y = np.sin(xn)
    fig = px.line(x=x,y=y)
    fig1 = px.line(x=x, y=fft(y).real)
    return fig, fig1

@my_app.callback(
    Output(component_id='output3', component_property='children'),
    [Input(component_id='drop3', component_property='value')]
)

def update_q3(input):
    return f'The selected item inside the dropdown menu is {input}'

my_app.run_server(port = 8080, host='0.0.0.0')