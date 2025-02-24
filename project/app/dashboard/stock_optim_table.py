import pandas as pd
from dash import html, dash_table
from project.config import download_from_gcs

def create_stock_optim(optim_df=None):
    if optim_df is None:
        blob_name = 'models/pulp_result/pulp_result_data.csv'
        pulp_result_file = download_from_gcs(blob_name)
        optim_df = pd.read_csv(pulp_result_file)
        optim_df = optim_df.drop(columns=['Unnamed: 0'])

    return html.Div([
        html.H3(
            "Optimization Recommendation:",
            style={
                'textAlign': 'left',
                'fontSize': '14px',
                'marginBottom': '10px',
                'fontFamily': 'Helvetica',
            }
        ),
        dash_table.DataTable(
            id='stock_optimization',
            columns=[
                {'name': col, 'id': col} for col in optim_df.columns
            ],
            data=optim_df.to_dict('records'),
            style_table={
                'width': '400px',
                'height': '250px',
                'overflowY': 'auto', 
                'overflowX': 'auto'
            },
            style_cell={
                'textAlign': 'left',
                'fontSize': '12px',
                'padding': '1px',
                'whiteSpace': 'normal',
                'wordBreak': 'break-word',
                'lineHeight': '12px',
                'fontFamily': 'Helvetica',
            },
            style_header={
                'backgroundColor': 'lightgrey', 
                'fontWeight': 'bold',
                'fontSize': '12px',
                'padding': '1px',
                'whiteSpace': 'normal',
                'wordBreak': 'break-word',
                'lineHeight': '12px'
            },
            fixed_rows={'headers': True},
            sort_action='native',
            sort_mode='multi',
            sort_by=[{'column_id': optim_df.columns[0], 'direction': 'asc'}],
            export_format='xlsx',
            export_headers='display',
        )
    ])