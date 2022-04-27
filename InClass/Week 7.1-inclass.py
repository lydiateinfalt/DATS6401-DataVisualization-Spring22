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

df_tesla = web.DataReader('TSLA', data_source = source, start= start_dt, end = end_dt)
df_apple = web.DataReader('AAPL', data_source = source, start= start_dt, end = end_dt)
df_oracle = web.DataReader('ORCL', data_source = source, start= start_dt, end = end_dt)
df_ibm = web.DataReader('IBM', data_source = source, start= start_dt, end = end_dt)
df_yelp = web.DataReader('YELP', data_source = source, start= start_dt, end = end_dt)
df_msft = web.DataReader('MSFT', data_source = source, start= start_dt, end = end_dt)

df1 = df_apple.copy()
df1['company'] = 'AAPL'

df2 = df_tesla.copy()
df2['company'] = 'TSLA'

df3 = df_oracle.copy()
df3['company'] = 'ORCL'

df4 = df_ibm.copy()
df4['company'] = 'IBM'

df5 = df_yelp.copy()
df5['company'] = 'YELP'

df6 = df_msft.copy()
df6['company'] = 'MSFT'


frames = [df1, df2, df3, df4, df5, df6]
result = pd.concat(frames)
#
# fig = px.line(x = df_tesla.index, y = df_tesla['Close'])
# # fig.show(renderer = 'browser')
fig = px.line(result, x = result.index, y = result.Close, color = 'company')
fig.show(renderer = 'browser')

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
# fig.add_trace(
#     go.Pie(values = result.Volume)
# )




