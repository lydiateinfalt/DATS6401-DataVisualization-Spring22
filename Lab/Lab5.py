import dash
import dash_html_components as html
import dash_core_components as dcc
import numpy as np
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWlwgP.css']

#Main
my_app = dash.Dash("My app", external_stylesheets= external_stylesheets)

#Question 1 Data -----------------------------------------------------------------------------------------


df = pd.read_csv("https://raw.githubusercontent.com/rjafari979/Complex-Data-Visualization-/main/CONVENIENT_global_confirmed_cases.csv")
df = df.dropna(axis=0, how='any')
columns = df.columns

df1 = df.copy()
df1 = df1.loc[1:]
df1['Date'] = pd.to_datetime(df1['Country/Region'])
countries = ['China', "United Kingdom", "Germany", "Brazil", "India", "Italy"]

for i in countries:
    col_names = df.columns[df.columns.str.startswith(i)]
    idx = [df.columns.get_loc(col) for col in col_names]
    # print(idx)
    new_column= i+"_sum"
    first_index = idx[0]
    last_index = idx[-1]
# china 57:89
# united Kingdom 249:259

df1['China_sum'] = df1.iloc[0:,57:90].astype(float).sum(axis=1)
df1['United Kingdom_sum'] = df1.iloc[:,249:260].astype(float).sum(axis=1)
df2=df1[['Date','US', 'Brazil', 'United Kingdom_sum', 'China_sum', 'India', 'Italy', 'Germany']]
#-------------------------------------------------------------------------------------------------
# Create a Dash layout
my_app.layout = html.Div([
    html.H1("Lab 5", style={'textAlign':'center'}),
    html.Br(),
    dcc.Tabs(id = 'hw-questions',
             children=[
                dcc.Tab(label = 'Question 1', value = 'q1'),
                dcc.Tab(label = 'Question 2', value = 'q2'),
                dcc.Tab(label = 'Question 3', value='q3'),
                dcc.Tab(label='Question 4', value='q4'),
                dcc.Tab(label='Question 5', value='q5'),
             ]),
    html.Div(id = 'layout')
])

tab1_layout = html.Div([
    html.H1('Question 1: Confirmed COVID Cases'),
    dcc.Dropdown(
        id='my-drop',
        options=[
            {'label': 'US', 'value': 'US'},
            {'label': 'Brazil', 'value': 'Brazil'},
            {'label': 'United Kingdom', 'value': 'United Kingdom_sum'},
            {'label': 'China', 'value': 'China_sum'},
            {'label': 'India', 'value': 'India'},
            {'label': 'Italy', 'value': 'Italy'},
            {'label': 'Germany', 'value': 'Germany'},
        ], clearable=False
    ),
    html.Br(),
    dcc.Graph(id = 'my-graph'),
])

tab2_layout = html.Div([
    html.H1('Question 2: Quadratic Equation'),
    html.P('Enter a:'),
    dcc.Input(id='input1', type="number"),
    html.P('Enter b:'),
    dcc.Input(id='input2', type='number'),
    html.P('Enter c:'),
    dcc.Input(id='input3', type='number'),
    dcc.Slider(id='n', min=1, max=1000, value=100, marks={10: '10', 50: '50', 100: '100', 500: '500', 1000: '1000'}),
    # html.Div(id='slider-output-container'),
    html.Br(),
    # html.Div(id = 'output2'),
    dcc.Graph(id = 'graph2'),
])

tab3_layout = html.Div([
    html.H1('Question 3: Calculator'),
    html.P('Enter first input a:'),
    dcc.Input(id='input4', type="number"),
    dcc.Dropdown(
        id = 'calc',
        options=[
            {'label': "+", 'value' : "add"},
            {'label': "-", 'value' : "subtract"},
            {'label': "*", 'value' : "multiply"},
            {'label': "/", 'value' : "divide"},
            {'label': "log", 'value' : "log"},
            {'label': "square", 'value' : "square"},
            {'label': "square root", 'value' : "squareroot"},
        ]
    ),
    html.P('Enter second input b:'),
    dcc.Input(id='input5', type='number'),
    html.Br(),
    html.Div(id = 'output3')
])

tab4_layout = html.Div([
    html.H1('Question 4 Gaussian Histogram'),
    html.P('Mean'),
    dcc.Slider(id='mean', min=-2, max=2, value=0,
               marks={-2: '-2', -1: '-1', 0: '0', 1: '1', 2: '2'}),

    html.Br(),
    html.P('STD'),
    dcc.Slider(id='std', min=1, max=3, value=1,
               marks={1: '1', 2: '2', 3: '3'}),

    html.Br(),
    html.P('Number of samples'),
    dcc.Slider(id='size', min=1, max=10000, value=500,
               marks={100: '100', 500: '500', 1000: '1000', 5000: '5000'}),

    html.Br(),
    html.P('Number of bins'),
    dcc.Dropdown(id='bins', options=[
        {'label': 20, 'value': 20},
        {'label': 30, 'value': 30},
        {'label': 40, 'value': 40},
        {'label': 60, 'value': 60},
        {'label': 80, 'value': 80},
        {'label': 100, 'value': 100}
    ], value=20, clearable=False),

    html.Br(),
    dcc.Graph(id = 'graph4'),
])

tab5_layout = html.Div([
    html.H1('Question 5: Polynomial Function'),
    html.P('Please enter polynomial order:'),
    dcc.Input(id='order', type="number"),
    html.Br(),
    dcc.Graph(id = 'poly-graph'),
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
    elif ques == 'q5':
        return tab5_layout

@my_app.callback(
    Output(component_id='my-graph', component_property='figure'),
    [Input(component_id='my-drop', component_property='value')]
)
def display_graph(country):
    return px.line(df2, x = 'Date', y = country)

@my_app.callback(
    Output(component_id='graph2', component_property='figure'),
    [Input(component_id='input1', component_property='value'),
     Input(component_id='input2', component_property='value'),
     Input(component_id='input3', component_property='value'),
     Input(component_id='n', component_property='value')]
)
def display_graph(a,b,c,n):
    # return f'a = {a}, b = {b}, c = {c}, n={n}'
    X=np.linspace(-2,2,num=n)
    return px.line(x=X,y=a*X**2 + b*X + c)

@my_app.callback(
    Output(component_id='output3', component_property='children'),
    [Input(component_id='input4', component_property='value'),
     Input(component_id='calc', component_property='value'),
     Input(component_id='input5', component_property='value'),]
)
def display_graph(a,calc,b):
    import math
    output = []
    if calc == 'add':
        return f'The output value is = {a + b}'
    elif calc == 'subtract':
        return f'The output value is = {a - b}'
    elif calc == 'multiply':
        return f'The output value is = {a * b}'
    elif calc == 'divide':
        return f'The output value is = {a / b}'
    elif calc == 'log':
        output = f'The output value for a is = {math.log(a)} \n'
        output +=f'The output value for b is = {math.log(b)}'
        return output
    elif calc == 'square':
        output =f'The output value for a is = {np.square(a)} \n'
        output += f'The output value for b is = {np.square(b)}'
        return output
    else:
        output = f'The output value for a is = {np.sqrt(a)}\n'
        output += f'The output value for b is = {np.sqrt(b)}'
        return output


@my_app.callback(
    Output(component_id='graph4', component_property='figure'),
    [Input(component_id='mean', component_property='value'),
     Input(component_id='std', component_property='value'),
     Input(component_id='size', component_property='value'),
     Input(component_id='bins', component_property='value'),]
)

def display_color(mean, std, size, bins):
    x = np.random.normal(mean, std, size = size)
    fig = px.histogram(x = x, nbins = bins, range_x=[-5, 5])
    return fig

@my_app.callback(
    Output(component_id='poly-graph', component_property='figure'),
    [Input(component_id='order', component_property='value')]
)
def display_poly(order):
    x = np.linspace(-2, 2)
    y = x**order
    return px.line(x,y)

my_app.run_server(port = 9090, host='0.0.0.0')
