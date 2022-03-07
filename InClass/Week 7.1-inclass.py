import plotly.express as px
import pandas_datareader as web
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# fig = px.line(x =[1,2,3], y = [1,2,3])
# fig.show(renderer = 'browser')
iris = px.data.iris()


stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']

#Start Date
start_dt = '2000-01-01'

#End Date
end_dt = '2021-09-18'

#Source to read in stocks data
source = 'yahoo'

#Create empty dataframe
df1 = pd.DataFrame()

for i in range(len(stocks)):
    df = web.DataReader(stocks[i], data_source = source, start= start_dt, end = end_dt)
    df['Symbol'] = stocks[i]
    df1 = df1.append(df)

frames = []
#
# fig = px.line(x = df_tesla.index, y = df_tesla['Close'])
fig = px.line(x = df1.index, y = df1['Close'])
#fig.show(renderer = 'browser')

#bar

fig = px.bar(iris, x = 'sepal_width', y = 'sepal_length')
#fig.show(renderer = 'browser')

fig = px.bar(iris, x = 'sepal_width', y = 'sepal_length', color='species',
             hover_data=['petal_width'])
# fig.show(renderer = 'browser')

fig = px.bar(iris, y = 'sepal_width', x = 'sepal_length', color='species',
             hover_data=['petal_width'], orientation='h')
# fig.show(renderer = 'browser')

#x = total bill y, day
tips = px.data.tips()
fig = px.bar(tips, y = 'day', x = 'total_bill',color='sex', barmode='stack',
             orientation='h')
#fig.show(renderer = 'browser')

#x = total bill y, day
tips = px.data.tips()
fig = px.bar(tips, y = 'day', x = 'total_bill',color='sex', barmode='group',
             orientation='h')
# fig.show(renderer = 'browser')

fig = px.pie(tips,
             values = 'total_bill',
             names = 'day')
fig.show(renderer = 'browser')

fig = px.box(tips, x = 'day', y = 'total_bill')
# fig.show(renderer = 'browser')

fig = px.violin(tips, x = 'day', y = 'total_bill')
# fig.show(renderer = 'browser')

fig = make_subplots(rows = 1, cols = 2)
fig.add_trace(
    go.Scatter(x=[1,2,3], y = [5,6,7])
)
fig.show(renderer = 'browser')

fig = make_subplots(rows = 1, cols = 2)
fig.add_trace(
    go.Pie(values = result.Volume)
)




