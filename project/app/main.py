# Entry point for the application
from dash import Dash, html, Input, Output
from flask import jsonify
from .dashboard import *
from .chatbot import *
from project.data.query import *
from project.models.pulp_solver import pulp_solver
import os
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# Get all necessary dfs
current_dir = os.getcwd()
processed_dir = 'project/data/processed/'
file_path1 = os.path.join(processed_dir, 'stock_pivot_data.csv')
file_path2 = os.path.join(processed_dir, 'pulp_result_data.csv')
file_path3 = os.path.join(processed_dir, 'metrics_raw_data.csv')

# Store the original data globally
if not os.path.exists(file_path1):
    get_stock_pivot_table()

if not os.path.exists(file_path2):
    get_pulp_raw()
    get_cost_table()  
    pulp_solver()

if not os.path.exists(file_path3):
    get_metrics_raw()

optim_df = pd.read_csv(os.path.join(processed_dir, 'pulp_result_data.csv'))
optim_df = optim_df.drop(columns=['Unnamed: 0'])
stock_pivot = pd.read_csv(os.path.join(processed_dir, 'stock_pivot_data.csv'))
stock_pivot = stock_pivot.drop(columns=['Unnamed: 0'])
metrics_raw_data = pd.read_csv(os.path.join(processed_dir, 'metrics_raw_data.csv'))
logging.info(f'File Ready! in {processed_dir}, current working directory: {current_dir}')

app = Dash(__name__)
server = app.server


#dashboard_layout = html.Div([
app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1(
                "STOCK OPTIMIZATION HUB DASHBOARD", 
                style={
                    'position': 'sticky',
                    'top':'0',
                    'textAlign': 'left',
                    'fontFamily': 'Helvetica',
                    'marginBottom': '20px',
                    'marginLeft': '20px',
                    'color': 'white',
                    'padding-left': '25px',
                    'zIndex':'1000',
                }
            ),
        ], style={
            'flex': '1',
            'width': '39%',
            'position': 'sticky',
            'zIndex':'1000',
        }),
        html.Div([
            product_filter(stock_pivot),
            from_filter(optim_df),
            to_filter(optim_df),
            html.Div([
                html.Button('Get new data', id='refresh-button', n_clicks=0, style={'margin' : '10px'}),
                html.Div(id='status-message'),
            ]),
        ], style={
            'position': 'sticky',
            'top': '0',
            'display': 'flex',          
            'alignItems': 'center',     
            'justifyContent': 'space-between',  
            'marginBottom': '2px',
            'width': '59%',
            'flex': '1',
            'zIndex':'1000',
        }),
    ], style={
        'position':'sticky',
        'top': '0',
        'display': 'flex',          
        'alignItems': 'center',  
        'marginBottom': '2px',
        'width': '100%',
        'backgroundColor': '#007BFF',
        'border': '1px solid #ddd', 
        'borderRadius': '10px',
        'zIndex': '1000',
    }),
    html.Div([
        dcc.Graph(id="stockout-ratio-scorecard", 
                figure=create_stockout_scorecard(metrics_raw_data)),
        dcc.Graph(id="stockout-ratio-linechart", 
                figure=create_stockout_linechart(metrics_raw_data)),
        dcc.Graph(id="stockout-ratio-barchart", 
                figure=create_stockout_barchart(metrics_raw_data)),
    ], style={
        'display': 'flex',          
        'alignItems': 'center',
        'marginBottom': '2px',
        'width': '100%',
        'justifyContent': 'space-evenly',
        'margin': 'auto'
    }),
    html.Div([
        dcc.Graph(id="itr-scorecard", 
                figure=create_itr_scorecard(metrics_raw_data)),
        dcc.Graph(id="itr-linechart", 
                figure=create_itr_linechart(metrics_raw_data)),
        dcc.Graph(id="itr-barchart", 
                figure=create_itr_barchart(metrics_raw_data)),
    ], style={
        'display': 'flex',          
        'alignItems': 'center',  
        'marginBottom': '2px',
        'width': '100%',
        'justifyContent': 'space-evenly',
        'margin': 'auto'      
    }),
    html.Div([
        dcc.Graph(id="overstock-cost-scorecard", 
                figure=create_overstock_cost_scorecard(metrics_raw_data)),
        dcc.Graph(id="overstock-cost-linechart", 
                figure=create_overstock_cost_linechart(metrics_raw_data)),
        dcc.Graph(id="overstock-cost-barchart", 
                figure=create_overstock_cost_barchart(metrics_raw_data)),
    ], style={
        'display': 'flex',          
        'alignItems': 'center',  
        'marginBottom': '2px',
        'width': '100%',
        'justifyContent': 'space-evenly',
        'margin': 'auto'           
    }),
    html.Div([
        dcc.Graph(id='forecast-vs-actual-linechart',
                  figure=create_fct_vs_act_linechart(metrics_raw_data), style={'width':'50%'}),
        create_stock_optim(optim_df),
        ], style={
        'display': 'flex',          
        'alignItems': 'center',  
        'marginBottom': '2px',
        'width': '100%',
        'justifyContent': 'center',
        }),
    create_table(stock_pivot),
    html.Div([
    # Tombol untuk membuka/meminimalkan chatbot
        html.Div(
            html.Button("💬 Chat", id="toggle-chat", style={"borderRadius": "50%", "width": "60px", "height": "60px", "backgroundColor": "#007BFF", "color": "white"}),
            style={"position": "fixed", "bottom": "20px", "right": "20px", "zIndex": "1000"}
        ),
        html.Div(id="chat-window", children = chatbox(),style={'display': 'none',"bottom": "90px", "right": "20px", "position": "fixed", "zIndex": "1000"}),
    ])
])

@app.callback(
    Output('stock_pivot','data'),
    Output('stock_optimization', 'data'),
    Input('product_filter_dropdown', 'value'),
    Input('from_filter_dropdown', 'value'),
    Input('to_filter_dropdown', 'value'),
)
def update_tables(selected_product_ids, 
                  selected_from_warehouse, 
                  selected_to_warehouse):
    
    global stock_pivot, optim_df
    filtered_pivot = stock_pivot.copy()
    filtered_optim = optim_df.copy()

    selected_products = selected_product_ids or filtered_pivot['Product_ID'].unique()
    selected_from = selected_from_warehouse or filtered_optim['From'].unique()
    selected_to = selected_to_warehouse or filtered_optim['To'].unique()
    selected_warehouse = list(set(selected_from).union(set(selected_to))) \
                            if selected_from_warehouse and selected_to_warehouse \
                            else selected_from_warehouse or selected_to_warehouse or filtered_optim['From'].unique()

    # Filter pivot table
    pivot_filter = filtered_pivot['Product_ID'].isin(selected_products)
    filtered_pivot = filtered_pivot[pivot_filter]

    selected_warehouse = [item for item in selected_warehouse if item != 'Supplier']
    columns = ['Product_ID', 'Category', 'Brand', 'Color']
    columns.extend(selected_warehouse or [])
    filtered_pivot = filtered_pivot[columns]
    
    # Filter filtered_optim 
    optim_filter = (filtered_optim['Product_ID'].isin(selected_products)) & \
                    (filtered_optim['From'].isin(selected_from)) & \
                    (filtered_optim['To'].isin(selected_to))
    filtered_optim = filtered_optim[optim_filter]

    return filtered_pivot.to_dict('records'), filtered_optim.to_dict('records')

@app.callback(
    Output('stockout-ratio-scorecard', 'figure'), 
    Output('itr-scorecard', 'figure'), 
    Output('overstock-cost-scorecard', 'figure'), 
    Output('stockout-ratio-linechart', 'figure'), 
    Output('itr-linechart', 'figure'), 
    Output('overstock-cost-linechart', 'figure'), 
    Output('stockout-ratio-barchart', 'figure'), 
    Output('itr-barchart', 'figure'), 
    Output('overstock-cost-barchart', 'figure'), 
    Output('forecast-vs-actual-linechart', 'figure'),
    Input('product_filter_dropdown', 'value'),
    Input('from_filter_dropdown', 'value'),
    Input('to_filter_dropdown', 'value')
)
def update_charts(selected_product_ids, 
                  selected_from_warehouse, 
                  selected_to_warehouse):
    
    global metrics_raw_data
    filtered_metrics = metrics_raw_data.copy()
    if selected_product_ids or selected_from_warehouse or selected_to_warehouse:
        selected_products = selected_product_ids or filtered_metrics['Product_ID'].unique()
        filtered_metrics = filtered_metrics[filtered_metrics['Product_ID'].isin(selected_products)]

        selected_from = selected_from_warehouse or filtered_metrics['Warehouse_Loc_ID'].unique()
        selected_to = selected_to_warehouse or filtered_metrics['Warehouse_Loc_ID'].unique()
        selected_warehouse = list(set(selected_from).union(set(selected_to))) \
                                if selected_from_warehouse and selected_to_warehouse \
                                else selected_from_warehouse or selected_to_warehouse or filtered_metrics['Warehouse_Loc_ID'].unique()
        filtered_metrics = filtered_metrics[filtered_metrics['Warehouse_Loc_ID'].isin(selected_warehouse)]
        
    return [
        create_stockout_scorecard(filtered_metrics),
        create_itr_scorecard(filtered_metrics),
        create_overstock_cost_scorecard(filtered_metrics),
        create_stockout_linechart(filtered_metrics),
        create_itr_linechart(filtered_metrics),
        create_overstock_cost_linechart(filtered_metrics),
        create_stockout_barchart(filtered_metrics),
        create_itr_barchart(filtered_metrics),
        create_overstock_cost_barchart(filtered_metrics),
        create_fct_vs_act_linechart(filtered_metrics)       
    ]

# Callback to show chatbot window
# Callback untuk menampilkan/menyembunyikan chatbot
@app.callback(
    Output("chat-window", "style"),
    Input("toggle-chat", "n_clicks"),
    State("chat-window", "style"),
    prevent_initial_call=True
)
def toggle_chat(n_clicks, current_style):
    if current_style["display"] == "none":
        return {**current_style, "display": "block"}
    return {**current_style, "display": "none"}

# Callback to refresh the pages
@app.callback(
    Output('status-message', 'children'),
    Input('refresh-button', 'n_clicks'),
    prevent_initial_call=True
)
def run_script_and_refresh(n_clicks):
    if n_clicks is None:
        return ""

    try:
        # Run the Python script
        subprocess.run(
            ["python", "-m", "project.data.hourly_pipeline"], 
            capture_output=True, text=True, check=True
        )
        # Return success message
        return "Complete! Please refresh page"
    
    except subprocess.CalledProcessError as e:
        # Return error message and trigger refresh
        return f"Error executing script:\n{e.stderr}"
    except Exception as ex:
        # Return unexpected error message and trigger refresh
        return f"Unexpected error: {ex}", 0

@server.route('/run-weekly-pipeline', methods=['GET'])
def run_weekly_pipeline():
    try:
        # Run the background task using subprocess
        subprocess.run(
            ["python", "-m", "project.data.weekly_pipeline"], 
            capture_output=True, text=True, check=True
        )
        # Return success response
        return jsonify({"status": "success", "message": "Weekly pipeline executed successfully."}), 200
    except subprocess.CalledProcessError as e:
        # Return error message
        return jsonify({"status": "error", "message": f"Error executing script: {e.stderr}"}), 500
    except Exception as ex:
        # Return unexpected error message
        return jsonify({"status": "error", "message": f"Unexpected error: {str(ex)}"}), 500

@server.route('/run-daily-pipeline', methods=['GET'])
def run_daily_pipeline():
    try:
        # Run the background task using subprocess
        subprocess.run(
            ["python", "-m", "project.data.weekly_pipeline"], 
            capture_output=True, text=True, check=True
        )
        # Return success response
        return jsonify({"status": "success", "message": "Daily pipeline executed successfully."}), 200
    except subprocess.CalledProcessError as e:
        # Return error message
        return jsonify({"status": "error", "message": f"Error executing script: {e.stderr}"}), 500
    except Exception as ex:
        # Return unexpected error message
        return jsonify({"status": "error", "message": f"Unexpected error: {str(ex)}"}), 500

@server.route('/run-hourly-pipeline', methods=['GET'])
def run_hourly_pipeline():
    try:
        # Run the background task using subprocess
        subprocess.run(
            ["python", "-m", "project.data.hourly_pipeline"], 
            capture_output=True, text=True, check=True
        )
        # Return success response
        return jsonify({"status": "success", "message": "Hourly pipeline executed successfully."}), 200
    except subprocess.CalledProcessError as e:
        # Return error message
        return jsonify({"status": "error", "message": f"Error executing script: {e.stderr}"}), 500
    except Exception as ex:
        # Return unexpected error message
        return jsonify({"status": "error", "message": f"Unexpected error: {str(ex)}"}), 500

if __name__ == "__main__":
    app.run_server(debug=False)