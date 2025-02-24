import pandas as pd
import plotly.graph_objects as go
from project.config import download_from_gcs

def create_stockout_scorecard(metrics_raw_data=None):
    # Define the project root and load the data
    if metrics_raw_data is None:
        blob_name = 'query_result/metrics_raw_data.csv'
        metrics_raw_data_file = download_from_gcs(blob_name)
        metrics_raw_data = pd.read_csv(metrics_raw_data_file)

    # Calculate Stockout Ratio per Year-Quarter
    stockout_ratio = metrics_raw_data.groupby('Year_Quarter', as_index=False).agg({'Daily_Sales':'sum', 'Lost_Sales':'sum'})
    stockout_ratio['stockout_ratio'] = (stockout_ratio['Lost_Sales'] / (stockout_ratio['Lost_Sales'] + stockout_ratio['Daily_Sales'])) * 100
    stockout_ratio.drop(columns=['Daily_Sales', 'Lost_Sales'], inplace=True)

    # Calculate Current
    stockout_yr_q = sorted(stockout_ratio['Year_Quarter'].unique())
    latest_stockout_ratio = stockout_ratio[stockout_ratio['Year_Quarter'] == stockout_ratio['Year_Quarter'].max()]['stockout_ratio'].values[0]
    latestmin1_stockout_ratio = stockout_ratio[stockout_ratio['Year_Quarter'] == stockout_yr_q[-2]]['stockout_ratio'].values[0]

    # Create the scorecard
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode='number+delta',
        value=latest_stockout_ratio,
        number={'suffix': '%'},
        delta={
            'position': 'bottom', 
            'reference': latestmin1_stockout_ratio,
            'relative': True,
            'valueformat':'.1%',
            'decreasing': {'color': 'green'},
            'increasing': {'color': 'red'},
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))

    fig.update_layout(
        template='plotly_white',
        title={
            'text': 'Stockout Ratio (Current Quarter)',
            'font':{'size':14},
            'x':0.5,
            'xanchor':'center',
            },
        height=180,
        width=320,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig

def create_stockout_linechart(metrics_raw_data=None):
    # Define the project root and load the data
    if metrics_raw_data is None:
        blob_name = 'query_result/metrics_raw_data.csv'
        metrics_raw_data_file = download_from_gcs(blob_name)
        metrics_raw_data = pd.read_csv(metrics_raw_data_file)

    # Calculate Stockout Ratio per Year-Quarter
    stockout_ratio = metrics_raw_data.groupby('Year_Quarter', as_index=False).agg({'Daily_Sales':'sum', 'Lost_Sales':'sum'})
    stockout_ratio['stockout_ratio'] = (stockout_ratio['Lost_Sales'] / (stockout_ratio['Lost_Sales'] + stockout_ratio['Daily_Sales'])) * 100
    stockout_ratio.drop(columns=['Daily_Sales', 'Lost_Sales'], inplace=True)

    # Create the area chart figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=stockout_ratio['Year_Quarter'],
        y=stockout_ratio['stockout_ratio'],
        mode='lines+markers+text',
        text=stockout_ratio['stockout_ratio'].apply(lambda x: f'{x:.1f}%'),
        textposition='bottom center',
        textfont=dict(size=9.5),
        name='Stockout Ratio',
        line=dict(color='royalblue'),
        fill='tozeroy'
    ))

    fig.update_layout(
        title={
            'text':'Stockout Ratio by Year-Quarter',
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

def create_stockout_barchart(metrics_raw_data=None):
    # Define the project root and load the data
    if metrics_raw_data is None:
        blob_name = 'query_result/metrics_raw_data.csv'
        metrics_raw_data_file = download_from_gcs(blob_name)
        metrics_raw_data = pd.read_csv(metrics_raw_data_file)

    # Calculate Stockout Ratio per Year-Quarter
    stockout_ratio = metrics_raw_data[metrics_raw_data['Year_Quarter']==metrics_raw_data['Year_Quarter'].max()]
    stockout_ratio = stockout_ratio.groupby('Product_ID', as_index=False).agg({'Daily_Sales':'sum', 'Lost_Sales':'sum'})
    stockout_ratio['stockout_ratio'] = (stockout_ratio['Lost_Sales'] / (stockout_ratio['Lost_Sales'] + stockout_ratio['Daily_Sales'])) * 100
    stockout_ratio.drop(columns=['Daily_Sales', 'Lost_Sales'], inplace=True)
    stockout_ratio = stockout_ratio.nlargest(5, 'stockout_ratio')

    # Create the bar chart figure
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=stockout_ratio['Product_ID'],
        y=stockout_ratio['stockout_ratio'],
        text=stockout_ratio['stockout_ratio'].apply(lambda x: f'{x:.1f}%'),
        textposition='auto',
        marker=dict(color='royalblue', line=dict(color='royalblue', width=1.5))
    ))

    fig.update_layout(
        title={
            'text':'Top 5 Stockout Ratio Product',
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