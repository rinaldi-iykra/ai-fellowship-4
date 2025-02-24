import pandas as pd
import plotly.graph_objects as go
from project.config import download_from_gcs

def create_fct_vs_act_linechart(metrics_raw_data=None):
    # Define the project root and load the data
    if metrics_raw_data is None:
        blob_name = 'query_result/metrics_raw_data.csv'
        metrics_raw_data_file = download_from_gcs(blob_name)
        metrics_raw_data = pd.read_csv(metrics_raw_data_file)

    # Calculate Stockout Ratio per Year-Quarter
    d_min30 = sorted(metrics_raw_data['Date'].unique().tolist())[-30:]
    fct_vs_act = metrics_raw_data[metrics_raw_data['Date'].isin(d_min30)]
    fct_vs_act = fct_vs_act.groupby('Date', as_index=False).agg({'Daily_Sales':'sum', 'Forecasted_Demand':'sum'})
    fct_vs_act = fct_vs_act.sort_values(by=['Date'])
    fct = fct_vs_act['Forecasted_Demand']
    act = fct_vs_act['Daily_Sales']

    fct_text = [None] * (len(fct) - 1) + [fct.iloc[-1]]
    fct_text = [f'{x:,.1f}' if x is not None else None for x in fct_text]
    act_text = [None] * (len(act) - 1) + [act.iloc[-1]]
    act_text = [f'{x:,.1f}' if x is not None else None for x in act_text]

    # Create the area chart figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fct_vs_act['Date'],
        y=fct_vs_act['Daily_Sales'],
        mode='lines+markers+text',
        text=act_text,
        textposition='bottom center',
        textfont=dict(size=9.5),
        name='Actual',
        line=dict(color='royalblue'),
    ))

    fig.add_trace(go.Scatter(
        x=fct_vs_act['Date'],
        y=fct_vs_act['Forecasted_Demand'],
        mode='lines+markers+text',
        text=fct_text,
        textposition='bottom center',
        textfont=dict(size=9.5),
        name='Forecast',
        line=dict(color='#FF6F61'),
    ))

    fig.update_layout(
        title={
            'text':'Forecast vs.Demand',
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
        showlegend=True,
        legend={
            'x':0,
            'y':1.1,
            'xanchor':'left',
            'yanchor':'top',
            'orientation':'h',
            'font':{'size':9},
        },
        template='plotly_white',
        height=300,
        width=600,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig


