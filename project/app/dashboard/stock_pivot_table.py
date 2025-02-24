import pandas as pd
from dash import html, dash_table
from project.config import download_from_gcs

def create_table(stock_pivot=None):
    if stock_pivot is None:
        blob_name = 'query_result/stock_pivot_data.csv'
        stock_pivot_file = download_from_gcs(blob_name)
        stock_pivot = pd.read_csv(stock_pivot_file)
        stock_pivot = stock_pivot.drop(columns=['Unnamed: 0'])

    return html.Div([
        html.H3(
            "Latest Stock Pivot Level per Warehouse", 
            style={
                'textAlign': 'left',
                'fontSize': '14px',
                'marginBottom': '10px',
                'fontFamily': 'Helvetica',
            }
        ),
        dash_table.DataTable(
            id='stock_pivot',
            columns=[
                {'name': col, 'id': col} for col in stock_pivot.columns
            ],
            data=stock_pivot.to_dict('records'), 
            style_table={
                'height': '300px', 
                'overflowY': 'auto', 
                'overflowX': 'auto'
            },
            style_cell={
                'textAlign': 'center',
                'fontSize': '11px',
                'padding': '0',
                'whiteSpace': 'normal',
                'wordBreak': 'break-word',
                'lineHeight': '12px',
                'fontFamily': 'Helvetica',
            },
            style_header={
                'backgroundColor': 'lightgrey',
                'fontWeight': 'bold',
                'fontSize': '12px',
                'padding': '0',
                'lineHeight': '12px',
                'height': 'auto'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': col},
                    'width': '15px'
                }
                for col in stock_pivot.columns
            ] + [
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f9f9f9'
                },
                {
                    'if': {'row_index': 'even'},
                    'backgroundColor': '#e0e0e0'
                }
            ],
            fixed_rows={'headers': True},
            sort_action='native',
            sort_mode='multi',
            sort_by=[{'column_id': stock_pivot.columns[0], 'direction': 'asc'}],
        )
    ])
