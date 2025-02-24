import pandas as pd
import plotly.graph_objects as go
from project.config import download_from_gcs

def create_overstock_cost_scorecard(metrics_raw_data=None):
    # Load the data if no external df is provided
    if metrics_raw_data is None:
        blob_name = 'query_result/metrics_raw_data.csv'
        metrics_raw_data_file = download_from_gcs(blob_name)
        metrics_raw_data = pd.read_csv(metrics_raw_data_file)
    
    metrics_raw_data['Overstock_Cost'] = (metrics_raw_data['Stock_Level'] - metrics_raw_data['Daily_Sales']) * metrics_raw_data['Inventory_Holding_Cost']
    overstock_cost = metrics_raw_data.groupby('Year_Quarter', as_index=False).agg({'Overstock_Cost':'sum'})
    overstock_cost['Overstock_Cost'] = overstock_cost['Overstock_Cost'] / 1000000000

    # Output latest overstock_cost data
    overstock_yr_q = sorted(overstock_cost['Year_Quarter'].unique())
    latest_overstock_cost = overstock_cost[overstock_cost['Year_Quarter'] == overstock_cost['Year_Quarter'].max()]['Overstock_Cost'].values[0]
    latestmin1_overstock_cost = overstock_cost[overstock_cost['Year_Quarter'] == overstock_yr_q[-2]]['Overstock_Cost'].values[0]

    # Create the scorecard figure
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode='number+delta',
        value=latest_overstock_cost,
        number={'valueformat': ',.0f', 'prefix': 'Rp', 'suffix': 'M'},
        delta={
            'position': 'bottom', 
            'reference': latestmin1_overstock_cost,
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
            'text': 'Overstock Cost (Current Quarter)',
            'font':{'size':14},
            'x':0.5,
            'xanchor':'center',
            },
        height=180,
        width=320,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig

def create_overstock_cost_linechart(metrics_raw_data=None):
    # Load the data if no external df is provided
    if metrics_raw_data is None:
        blob_name = 'query_result/metrics_raw_data.csv'
        metrics_raw_data_file = download_from_gcs(blob_name)
        metrics_raw_data = pd.read_csv(metrics_raw_data_file)
    
    metrics_raw_data['Overstock_Cost'] = (metrics_raw_data['Stock_Level'] - metrics_raw_data['Daily_Sales']) * metrics_raw_data['Inventory_Holding_Cost']
    overstock_cost = metrics_raw_data.groupby('Year_Quarter', as_index=False).agg({'Overstock_Cost':'sum'})
    overstock_cost['Overstock_Cost'] = overstock_cost['Overstock_Cost'] / 1000000000

    # Create the line chart figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=overstock_cost['Year_Quarter'],
        y=overstock_cost['Overstock_Cost'],
        mode='lines+markers+text',
        text=overstock_cost['Overstock_Cost'].apply(lambda x: f'{x:,.0f} M'),
        textposition='top center',
        name='Overstock Cost',
        line=dict(color='skyblue'),
        fill='tozeroy'
    ))

    fig.update_layout(
        title={
            'text':'Overstock Cost by Year-Quarter',
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

def create_overstock_cost_barchart(metrics_raw_data=None):
    # Load the data if no external df is provided
    if metrics_raw_data is None:
        blob_name = 'query_result/metrics_raw_data.csv'
        metrics_raw_data_file = download_from_gcs(blob_name)
        metrics_raw_data = pd.read_csv(metrics_raw_data_file)
    
    metrics_raw_data = metrics_raw_data[metrics_raw_data['Year_Quarter']==metrics_raw_data['Year_Quarter'].max()]
    metrics_raw_data['Overstock_Cost'] = (metrics_raw_data['Stock_Level'] - metrics_raw_data['Daily_Sales']) * metrics_raw_data['Inventory_Holding_Cost']
    overstock_cost = metrics_raw_data.groupby('Product_ID', as_index=False).agg({'Overstock_Cost':'sum'})
    overstock_cost['Overstock_Cost'] = overstock_cost['Overstock_Cost'] / 1000000000
    overstock_cost = overstock_cost.nlargest(5, 'Overstock_Cost')

    # Create the line chart figure
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=overstock_cost['Product_ID'],
        y=overstock_cost['Overstock_Cost'],
        text=overstock_cost['Overstock_Cost'].apply(lambda x:f'{x:,.1f} M'),
        textposition='auto',
        marker=dict(color='skyblue', line=dict(color='skyblue', width=1.5))
    ))

    fig.update_layout(
        title={
            'text':'Top 5 Overstock Cost Product',
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