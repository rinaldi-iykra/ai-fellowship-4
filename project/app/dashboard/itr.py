import pandas as pd
import plotly.graph_objects as go
from project.config import download_from_gcs

def create_itr_scorecard(metrics_raw_data=None):
    # Define the project root and load the data
    if metrics_raw_data is None:
        blob_name = 'query_result/metrics_raw_data.csv'
        metrics_raw_data_file = download_from_gcs(blob_name)
        metrics_raw_data = pd.read_csv(metrics_raw_data_file)

    # Further preprocess the data
    metrics_raw_data['Date'] = pd.to_datetime(metrics_raw_data['Date'])
    itr = metrics_raw_data.groupby('Year_Quarter', as_index=False).agg({'Date':['min','max'], 'Daily_Sales':'sum'})
    itr.columns = ['_'.join(col).strip() if col[1] else col[0] for col in itr.columns.values]

    earliest_stock_sum = [(metrics_raw_data[metrics_raw_data['Date']==i]['Stock_Level'].sum()) for i in itr['Date_min']]
    latest_stock_sum = [(metrics_raw_data[metrics_raw_data['Date']==i]['Stock_Level'].sum()) for i in itr['Date_max']]

    itr['Earliest_Stock_Sum'] = earliest_stock_sum
    itr['Latest_Stock_Sum'] = latest_stock_sum

    itr['ITR'] = (itr['Daily_Sales_sum']/((itr['Latest_Stock_Sum']+itr['Earliest_Stock_Sum'])/2)).round(2)
    itr.drop(columns=['Date_min','Date_max', 'Daily_Sales_sum', 'Earliest_Stock_Sum', 'Latest_Stock_Sum'], inplace=True)

    # Output latest ITR data
    itr_yr_q = sorted(itr['Year_Quarter'].unique())
    latest_itr = itr[itr['Year_Quarter'] == itr['Year_Quarter'].max()]['ITR'].values[0]
    latestmin1_itr = itr[itr['Year_Quarter'] == itr_yr_q[-2]]['ITR'].values[0]

    # Create the scorecard figure
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=latest_itr,
        delta={
            'position': "bottom", 
            'reference': latestmin1_itr,
            'relative': True,
            'valueformat': '.1%',
            'decreasing': {'color': 'green'},
            'increasing': {'color': 'red'},
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))

    fig.update_layout(
        template='plotly_white',
        title={
            'text': 'ITR (Current Quarter)',
            'font':{'size':14},
            'x':0.5,
            'xanchor':'center',
            },
        height=180,
        width=320,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig

def create_itr_linechart(metrics_raw_data=None):
    # Define the project root and load the data
    if metrics_raw_data is None:
        blob_name = 'query_result/metrics_raw_data.csv'
        metrics_raw_data_file = download_from_gcs(blob_name)
        metrics_raw_data = pd.read_csv(metrics_raw_data_file)

    metrics_raw_data['Date'] = pd.to_datetime(metrics_raw_data['Date'])
    itr = metrics_raw_data.groupby('Year_Quarter', as_index=False).agg({'Date':['min','max'], 'Daily_Sales':'sum'})
    itr.columns = ['_'.join(col).strip() if col[1] else col[0] for col in itr.columns.values]

    earliest_stock_sum = [(metrics_raw_data[metrics_raw_data['Date']==i]['Stock_Level'].sum()) for i in itr['Date_min']]
    latest_stock_sum = [(metrics_raw_data[metrics_raw_data['Date']==i]['Stock_Level'].sum()) for i in itr['Date_max']]

    itr['Earliest_Stock_Sum'] = earliest_stock_sum
    itr['Latest_Stock_Sum'] = latest_stock_sum

    itr['ITR'] = (itr['Daily_Sales_sum']/((itr['Latest_Stock_Sum']+itr['Earliest_Stock_Sum'])/2)).round(2)
    itr.drop(columns=['Date_min','Date_max', 'Daily_Sales_sum', 'Earliest_Stock_Sum', 'Latest_Stock_Sum'], inplace=True)

    # Create the line chart figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=itr['Year_Quarter'],
        y=itr['ITR'],
        mode='lines+markers+text',
        text=itr['ITR'],
        textposition='bottom center',
        name='ITR',
        line=dict(color='#483D8B'),
        fill='tozeroy'
    ))

    fig.update_layout(
        title={
            'text':'ITR by Year (Quarter)',
            'font':{'size':14},
            'x':0.5,
            'xanchor':'center',           
            },
        xaxis={
            'tickfont':{'size':13}
        },
        yaxis = {
            'tickfont':{'size':13},
        },
        template='plotly_white',
        height=250,
        width=500,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig

def create_itr_barchart(metrics_raw_data=None):
    # Define the project root and load the data
    if metrics_raw_data is None:
        blob_name = 'query_result/metrics_raw_data.csv'
        metrics_raw_data_file = download_from_gcs(blob_name)
        metrics_raw_data = pd.read_csv(metrics_raw_data_file)

    metrics_raw_data = metrics_raw_data[metrics_raw_data['Year_Quarter']==metrics_raw_data['Year_Quarter'].max()].copy()
    metrics_raw_data['Date'] = pd.to_datetime(metrics_raw_data['Date'])
    itr = metrics_raw_data.groupby(['Year_Quarter', 'Product_ID'], as_index=False).agg({'Date':['min','max'], 'Daily_Sales':'sum'})
    itr.columns = ['_'.join(col).strip() if col[1] else col[0] for col in itr.columns.values]

    itr_date = itr['Date_min'].unique().tolist()
    itr_date.append(itr['Date_max'].unique()[0])
    sum_stock = metrics_raw_data[metrics_raw_data['Date'].isin(itr_date)].copy()
    sum_stock = sum_stock.groupby(['Date', 'Product_ID'], as_index=False).agg({'Stock_Level':'sum'})

    itr = itr.merge(sum_stock, left_on=['Product_ID', 'Date_min'], right_on=['Product_ID', 'Date'], how='left')
    itr = itr.merge(sum_stock, left_on=['Product_ID', 'Date_max'], right_on=['Product_ID', 'Date'], how='left', suffixes=['-min','-max'])

    itr = itr.drop(columns=['Date-min', 'Date-max'])

    itr['ITR'] = (itr['Daily_Sales_sum']/((itr['Stock_Level-max']+itr['Stock_Level-min'])/2)).round(2)
    itr = itr.nlargest(5, 'ITR')
    itr.drop(columns=['Date_min','Date_max', 'Daily_Sales_sum', 'Stock_Level-min', 'Stock_Level-max'], inplace=True)

    # Create the bar chart figure
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=itr['Product_ID'],
        y=itr['ITR'],
        text=itr['ITR'],
        textposition='auto',
        marker=dict(color='#483D8B', line=dict(color='#483D8B', width=1.5))
    ))

    fig.update_layout(
        title={
            'text':'Top 5 Highest ITR Product',
            'font':{'size':14},
            'x':0.5,
            'xanchor':'center',
            },
        xaxis={
            'tickfont':{'size':13}
        },
        yaxis = {
            'tickfont':{'size':13},
        },
        template='plotly_white',
        height=250,
        width=500,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig