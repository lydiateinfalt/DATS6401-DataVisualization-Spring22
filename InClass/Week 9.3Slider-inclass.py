import dash as dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWlwgP.css']
my_app = dash.Dash("My app", external_stylesheets= external_stylesheets)

my_app.layout = html.Div([
    dcc.Slider(
        id = 'my-slider',
        min = 0,
        max = 20,
        step= 1,
        value= 10,

    ),
html.Div(id='slider-output-container')

])

@my_app.callback(
    Output(component_id='slider-output-container', component_property='children'),
    [Input(component_id='my-slider', component_property='value')]
)

def update_lydia(input):
    return f'You have selected {input}'

my_app.server.run(
    port = 8050,
    host='0.0.0.0'
)