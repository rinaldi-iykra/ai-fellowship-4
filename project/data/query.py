# Query BigQuery and store raw data
import pandas as pd
from project.config import *
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# Query to get pivot of latest stock
# Step 1: Collect distinct warehouse_loc_ids (concatenation of warehouse_loc and warehouse_id) into an array
def get_warehouse_ids():
    query_warehouse_ids = f"""
    SELECT ARRAY_AGG(DISTINCT CONCAT(Warehouse_Location, '-', Warehouse_ID) ORDER BY CONCAT(Warehouse_Location, '-', Warehouse_ID) ASC) AS warehouse_loc_ids
    FROM `{test_table_path}`
    """
    warehouse_loc_ids = client.query(query_warehouse_ids).result().to_dataframe().iloc[0, 0]
    return warehouse_loc_ids

# Step 2: Get the max date
def get_max_date():
    query_max_date = f"""
    SELECT MAX(Date) AS max_date
    FROM `{test_table_path}`
    """
    max_date = client.query(query_max_date).result().to_dataframe().iloc[0, 0]
    return max_date

# Step 3: Get list of warehouse_loc_ids
def get_warehouse_loc_ids_str():
    warehouse_loc_ids = get_warehouse_ids()
    warehouse_loc_ids_str = ', '.join([f"'{item}'" for item in warehouse_loc_ids])
    return warehouse_loc_ids_str

# Step 4: Build dynamic PIVOT query
def get_stock_pivot_table():
    logging.info('Get stock_pivot_data')
    max_date = get_max_date()
    warehouse_loc_ids_str = get_warehouse_loc_ids_str()

    stock_pivot_table_query = f"""
    SELECT * FROM (
        SELECT Product_ID, Category, Brand, Color,
        CONCAT(Warehouse_Location, '-', Warehouse_ID) AS Warehouse_Loc_ID, IFNULL(Stock_Level, 0) AS Stock_Level
        FROM `{test_table_path}`
        WHERE Date = '{max_date}'
    ) PIVOT (
        SUM(IFNULL(Stock_Level, 0)) FOR Warehouse_Loc_ID IN ({warehouse_loc_ids_str})
    )
    """
    df = client.query(stock_pivot_table_query).result().to_dataframe()
    df.to_csv('project/data/processed/stock_pivot_data.csv')
    logging.info('Get stock_pivot_data: success!')

# Query Item_transfer cost data
def get_cost_table():
    logging.info('Get cost_table')
    cost_table_query = f"""
    SELECT * FROM `{cost_table_path}`
    """
    df = client.query(cost_table_query).result().to_dataframe()
    upload_to_gcs(df, 'query_result/cost_table_data.csv')
    logging.info('Get cost_table: success!')

# Query the raw data for pulp
def get_pulp_raw():
    logging.info('Get pulp_raw_data')
    pulp_raw_query = f"""
    SELECT 
        Product_ID, 
        CONCAT(Warehouse_Location, '-', Warehouse_ID) as Warehouse_Loc_ID,
        Stock_Level,
        Reorder_Threshold,
        Reorder_Quantity, 
        Forecasted_Demand,
        Lead_Time,
        Batch_Size, 
        Supply_Cost_Per_Unit
    FROM `{test_table_path}`
    WHERE Date = (SELECT MAX(Date) FROM `{test_table_path}`)
    """
    df = client.query(pulp_raw_query).result().to_dataframe()
    upload_to_gcs(df, 'query_result/pulp_raw_data.csv')
    logging.info('Get pulp_raw_data: success!')

def get_metrics_raw():
    logging.info('Get metrics_raw_data')
    metrics_raw_query = f"""
    SELECT 
        Date, 
        EXTRACT(YEAR FROM Date) as Year, 
        EXTRACT(QUARTER FROM Date) as Q,
        EXTRACT(MONTH FROM Date) as Month,
        EXTRACT(WEEK FROM Date) as Week,
        CONCAT(EXTRACT(YEAR FROM Date), '-Q', EXTRACT(QUARTER FROM Date)) as Year_Quarter,
        Product_ID, 
        CONCAT(Warehouse_Location, '-', Warehouse_ID) as Warehouse_Loc_ID, 
        Daily_Sales,
        Forecasted_Demand, 
        Stock_Level,
        Inventory_Holding_Cost,
        Lost_Sales
    FROM `{test_table_path}`
    WHERE EXTRACT(YEAR FROM Date) BETWEEN 2023 AND 2024
    """
    df = client.query(metrics_raw_query).result().to_dataframe()
    df.to_csv('project/data/processed/metrics_raw_data.csv')
    logging.info('Get metrics_raw_data: success!')