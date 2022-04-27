#Lydia Teinfalt
#Quiz2
#04/06/2022
import dash
import plotly.data
from dash import html
import dash_core_components as dcc
from dash import dcc
import numpy as np
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#Main
my_app = dash.Dash("My app", external_stylesheets= external_stylesheets)
data = plotly.data.tips()

my_app.layout = html.Div([
    html.H3("Quiz 2"),
    html.P('Please select the feature from the menu'),
    dcc.Dropdown(
        id = 'my-drop',
        options=[
            {'label': 'Day', 'value': 'day'},
            {'label': 'Time', 'value': 'time'},
            {'label': 'Sex', 'value': 'sex'},
        ], clearable = False
    ),
    html.Br(),
    dcc.Dropdown(
        id='my-drop2',
        options=[
            {'label': 'total_bill', 'value': 'total_bill'},
            {'label': 'tip', 'value': 'tip'},
            {'label': 'size', 'value': 'size'},
        ], clearable=False
    ),
    html.Br(),
    dcc.Graph(id = 'my-graph'),

])

@my_app.callback(
    Output(component_id='my-graph', component_property='figure'),
    [Input(component_id='my-drop', component_property='value'),
     Input(component_id='my-drop2', component_property='value')]
)
def update_lydia(input1, input2):
    fig = px.pie(data, values=input2, names = input1, title="Pie Plot")
    return fig

my_app.server.run(
    port = 8050,
    host='0.0.0.0'
)