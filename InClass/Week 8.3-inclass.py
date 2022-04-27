import dash as dash
from dash import dcc
from dash import html

my_app = dash.Dash('My app')
my_app.layout = html.Div([
    html.Div(html.H1('Hello World with html.H1')),
    html.Div(html.H2('Hello World with html.H2')),
    html.Div(html.H3('Hello World with html.H3')),
    html.Div(html.H4('Hello World with html.H4')),
    html.Div(html.H5('Hello World with html.H5')),
    html.Div(html.H6('Hello World with html.H6')),
])

my_app.run_server(
    port = 8500,
    host = '0.0.0.0'
)