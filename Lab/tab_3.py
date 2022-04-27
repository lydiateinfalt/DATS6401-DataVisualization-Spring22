import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWlwgP.css']


df = pd.read_csv("https://raw.githubusercontent.com/rjafari979/Complex-Data-Visualization-/main/CONVENIENT_global_confirmed_cases.csv")
df1 = df.iloc[0]
dfnull = df.iloc[1:-1].isnull().values.any()
if dfnull:
    df = df.dropna(df.iloc[1:-1].isnull(), axis=0, inplace=True)

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

df1['China_sum'] = df1.iloc[:,57:89].astype(float).sum(axis=1)
df1['United Kingdom_sum'] = df1.iloc[:,249:259].astype(float).sum(axis=1)
df2=df1[['Date','US', 'Brazil', 'United Kingdom_sum', 'China_sum', 'India', 'Italy', 'Germany']]

my_app = dash.Dash("My app")

my_app.layout = html.Div([
            html.H3("Complex Data Vis"),
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
            dcc.Graph(id='my-graph'),
        ])

@my_app.callback(
    Output(component_id='my-graph', component_property='figure'),
    [Input(component_id='my-drop', component_property='value')]
)

def display_data(input):
    fig = px.line(df2, x = 'Date', y = input)
    return fig
#
my_app.server.run(
    port = 8050,
    host='0.0.0.0'
)