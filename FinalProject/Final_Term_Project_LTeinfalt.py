#Lydia Teinfalt
#Lab 5
#04/06/2022
import dash
from dash import html
from dash import dcc
import numpy as np
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import math as m

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
    html.H1("Lab 5", style={'textAlign':'center'}),
    html.Br(),
    dcc.Tabs(id = 'hw-questions',
             children=[
                dcc.Tab(label = 'Line Plots', value = 'q1'),
                dcc.Tab(label = 'Question 2', value = 'q2'),
                dcc.Tab(label = 'Question 3', value='q3'),
                dcc.Tab(label='Question 4', value='q4'),
                dcc.Tab(label='Question 5', value='q5'),
                dcc.Tab(label='Question 6', value='q6'),
             ]),
    html.Div(id = 'layout')
])

tab1_layout = html.Div([
    html.H1('Mushroom Line Plot'),
    html.H3("Pick variable to plot against stem height"),
    dcc.Dropdown(
        id='my-drop',
        options=[
            {'label': 'Cap Shape', 'value': 'cap-shape'},
            {'label': 'Cap Color', 'value': 'cap-color'},
            {'label': 'Gill Color', 'value': 'gill-color'},
        ], clearable=False
    ),
    html.Br(),
    dcc.Graph(id = 'my-graph'),
])

tab2_layout = html.Div([
    html.H1('Question 2: Quadratic Equation 𝑓(𝑥)=𝑎𝑥^2+𝑏𝑥+𝑐'),
    html.P('Select value for a:'),
    dcc.Slider(id = "input1",min = -2, max = 2, value = 1, marks = {-2:'-2',-1: '-1', 0: '0', 1:'1', 2:'2'}),
    html.P('Select value for b:'),
    dcc.Slider(id = "input2",min = -2, max = 2, value = 1, marks = {-2:'-2',-1: '-1', 0: '0', 1:'1', 2:'2'}),
    html.P('Select value for c:'),
    dcc.Slider(id = "input3",min = -2, max = 2, value = 1, marks = {-2:'-2',-1: '-1', 0: '0', 1:'1', 2:'2'}),
    html.P('Select value for number of samples:'),
    dcc.Slider(id='n', min=1, max=1000, value=100, marks={10: '10', 50: '50', 100: '100', 500: '500', 1000: '1000'}),
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
    dcc.Input(id='input5', type='number', value=1),
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

df_bar = pd.DataFrame({
"Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges","Bananas"],
"Amount": [4, 1, 2, 2, 4, 5],
"City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})
fig6 = px.bar(df_bar, x="Fruit", y="Amount", color="City", barmode="group")
tab6_layout = html.Div(children=[
    html.Div([
        html.Div([
            html.H1("Hello Dash 1"),
            dcc.Graph(
                id='graph61',
                figure=fig6
            ),
            html.H5("Slider 1"),
            dcc.Slider(id='n', min=0, max=20, value=1),
        ], className='six columns'),
        html.Div([
            html.H1("Hello Dash 2"),

            dcc.Graph(
                id='graph62',
                figure=fig6
            ),
            html.H5("Slider 2"),
            dcc.Slider(id='n', min=0, max=20, value=1),
        ], className='six columns'),
    ], className='row'),
    # New Div for all elements in the new 'row' of the page
    html.Div([
        html.Div([
            html.H1("Hello Dash 3"),
            dcc.Graph(
                id='graph63',
                figure=fig6
            ),
            html.H5("Slider 3"),
            dcc.Slider(id='n', min=0, max=20, value=1),
        ], className='six columns'),
        html.Div([
            html.H1("Hello Dash 4"),
            dcc.Graph(
                id='graph64',
                figure=fig6
            ),
            html.H5("Slider 4"),
            dcc.Slider(id='n', min=0, max=20, value=1),
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
    elif ques == 'q5':
        return tab5_layout
    else:
        return tab6_layout

@my_app.callback(
    Output(component_id='my-graph', component_property='figure'),
    [Input(component_id='my-drop', component_property='value')]
)
def display_graph(x):
    return px.line(df, x = x, y = 'stem-height')

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
def display_graph1(a,calc,b):
    import math
    output = []
    if calc == 'add':
        x = a + b
        return f'The result is: {x}'
    if calc == 'subtract':
        x = a - b
        return f'The result is: {x}'
    if calc == 'multiply':
        x = a * b
        return f'The result is: {x}'
    if calc == 'divide':
        x = a/b
        return f'The result is: {x}'
    if calc == 'log':
        x = math.log(a)
        return f'The result is: {x}'
    if calc == 'square':
        x = a ** 2
        return f'The result is: {x}'
    if calc == 'squareroot':
        x = a ** 0.5
        return f'The result is: {x}'

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

@my_app.callback(
    [Output(component_id='graph61', component_property='figure'),
    Output(component_id='graph62', component_property='figure'),
    Output(component_id='graph63', component_property='figure'),
    Output(component_id='graph64', component_property='figure')]
)
def display_dashboard():
    return ""


my_app.run_server(port = 8080, host='0.0.0.0')
