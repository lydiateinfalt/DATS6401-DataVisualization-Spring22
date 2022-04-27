import dash as dash
from dash import dcc
from dash import html
from dash.dependencies import Input,Output
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWlwgP.css']
app = dash.Dash("My app", external_stylesheets= external_stylesheets)

app.layout = html.Div([
    html.H1("Homework 4", style={'textAlign':'center'}),
    html.Br(),
    dcc.Tabs(id = 'hw-questions',
             children=[
                dcc.Tab(label = 'Question1', value = 'q1'),
                dcc.Tab(label = 'Question2', value = 'q2'),
                dcc.Tab(label = 'Question3', value='q3'),
             ]),
    html.Div(id = 'layout')
])

q1_layout = html.Div([
    html.H1('Question1'),
    html.H5('Test'),
    html.P('Enter Input:'),
    dcc.Input(id = 'input1', type = 'text' ),
    html.Br(),
    html.Div(id='my-out')
])

q2_layout = html.Div([
    html.H1('Complex Data Visualization'),
    dcc.Dropdown(id = 'drop3',
                 options = [
                    {'label': 'Introduction', 'value': 'Introduction'},
                    {'label': 'Panda Package', 'value': 'Panda Package'},
                 ], value = 'Introduction'),
    html.Br(),
    html.Div(id = 'output2')
])

q3_layout = html.Div([
    html.H1('Time series analysis'),
    dcc.Checklist(id = 'my-checklist',
                 options = [
                    {'label': 'ACF', 'value': 'ACF'},
                    {'label': 'GPAC', 'value': 'GPAC'},
                    {'label': 'Correlation', 'value': 'Correlation'},
                 ], value = ''),
    html.Br(),
    html.Div(id = 'output3')
])

@app.callback(Output(component_id='layout', component_property='children'),
            [Input(component_id='hw-questions', component_property='value')])
def update_layout(ques):
    if ques == 'q1':
        return q1_layout
    elif ques == 'q2':
        return q2_layout
    elif ques == 'q3':
        return q3_layout

@app.callback(
    Output(component_id='my-out', component_property='children'),
    [Input(component_id='input1', component_property='value')]
)

def update_lydia(input):
    return f'The entered item is {input}'

app.run_server(port = 9595, host='0.0.0.0')
