import dash as dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWlwgP.css']
my_app = dash.Dash("My app", external_stylesheets= external_stylesheets)

my_app.layout = html.Div([
    html.H3("Complex Data Vis"),
    dcc.Dropdown(
        id = 'my-drop',
        options=[
            {'label': 'Introduction', 'value': 'Introduction'},
            {'label': 'Panda', 'value': 'Panda'},
            {'label': 'Seaborn', 'value': 'Seaborn'},
            {'label': 'Matplotlib', 'value': 'Matplotlib'},
        ], clearable = False
    ),

html.Br(),
html.Div(id = 'my-out')

])

@my_app.callback(
    Output(component_id='my-out', component_property='children'),
    [Input(component_id='my-drop', component_property='value')]
)

def update_lydia(input):
    return f'The selected item is {input}'

my_app.server.run(
    port = 8050,
    host='0.0.0.0'
)