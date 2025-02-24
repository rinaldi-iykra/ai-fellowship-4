import numpy as np
import pandas as pd
from pulp import *
from project.config import *
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

def pulp_solver():
    # Get the pulp_raw_data from data/query_result for pulp model
    logging.info('Running the PuLP..')
    pulp_raw_df = pd.read_csv(download_from_gcs('query_result/pulp_raw_data.csv'))
    pulp_raw_df['Reorder_Threshold'] = pulp_raw_df['Forecasted_Demand'] * pulp_raw_df['Lead_Time']
    pulp_final_column = pulp_raw_df.columns.tolist()

    # Get the cost_table_data from data/query_result for pulp model
    cost_df = pd.read_csv(download_from_gcs('query_result/cost_table_data.csv'))

    # Initialize Model
    stock_optim_model = LpProblem("Minimize_Transportation_Cost", LpMinimize)

    # Prepare data needed for decision variable, constraint and objective function
    product = pulp_raw_df["Product_ID"].unique().tolist()
    origin = pulp_raw_df["Warehouse_Loc_ID"].unique().tolist()
    origin.append("Supplier")
    destination = pulp_raw_df["Warehouse_Loc_ID"].unique().tolist()

    origin_supply = {
        (p, o): pulp_raw_df[(pulp_raw_df["Product_ID"]==p) & (pulp_raw_df["Warehouse_Loc_ID"]==o)]["Stock_Level"].iloc[0]
        if o != "Supplier" else 10000 for p in product for o in origin
        }

    destination_demand = {
        (p, d): pulp_raw_df[(pulp_raw_df["Product_ID"]==p) & (pulp_raw_df["Warehouse_Loc_ID"]==d)]["Reorder_Threshold"].iloc[0]
        for p in product for d in destination
        }

    cost = {
        (p, o, d): (
            pulp_raw_df[
                (pulp_raw_df["Product_ID"] == p) &
                (pulp_raw_df["Warehouse_Loc_ID"] == d)
            ]["Supply_Cost_Per_Unit"].iloc[0]
            if o == "Supplier"
            else cost_df[
                (cost_df["Warehouse_Origin"] == o) &
                (cost_df["Warehouse_Destination"] == d)
            ]["Cost"].iloc[0]
            if o != d
            else 0
        )
        for p in product
        for o in origin
        for d in destination
    }


    # Initialize the linear programming problem
    stock_optim_model = LpProblem("Minimize_Transportation_Cost", LpMinimize)

    # Decision variables
    transfer_qty = {
        (p, o, d): LpVariable(f"Transfer_{p}_{o}_{d}", lowBound=0)
        for p in product for o in origin for d in destination
    }

    # Objective function
    stock_optim_model += lpSum(cost[p, o, d] * transfer_qty[p, o, d] for p in product for o in origin for d in destination), "Total_Cost"

    # Supply constraints
    for p in product:
        for o in origin:
            stock_optim_model += (
                lpSum(transfer_qty[p, o, d] for d in destination) <= origin_supply[p, o],
                f"Supply_Constraint_{p}_{o}"
            )

    # Demand constraints
    for p in product:
        for d in destination:
            stock_optim_model += (
                lpSum(transfer_qty[p, o, d] for o in origin) >= destination_demand[p, d],
                f"Demand_Constraint_{p}_{d}"
            )

    # Solve the problem
    stock_optim_model.solve()

    optim = []
    for p, o, d in transfer_qty:
        if o!=d and transfer_qty[p, o, d].varValue > 0:
            qty = int(transfer_qty[p, o, d].varValue)
            optim.append([o, d, p, qty])

    optim_df = pd.DataFrame(optim, columns=["From", "To", "Product_ID", "trfQty"])
    optim_df = optim_df.merge(pulp_raw_df, left_on=["Product_ID", "To"],
                            right_on=["Product_ID", "Warehouse_Loc_ID"], how="left")
    optim_df = optim_df[["From", "To", "Product_ID", "Reorder_Threshold", "trfQty"]]
    optim_df.rename(columns={"Reorder_Threshold":"Demand"}, inplace=True)

    # Save df
    optim_df.to_csv('project/data/processed/pulp_result_data.csv')

    # Transform data: RoQ should higher than Batch_Size
    roq_df = optim_df[optim_df['From']=='Supplier']
    roq_df_merged = roq_df.merge(pulp_raw_df[['Product_ID', 'Warehouse_Loc_ID', 'Batch_Size']], 
                                left_on=['To', 'Product_ID'],
                                right_on=['Warehouse_Loc_ID', 'Product_ID'],
                                how='left')
    roq_df_merged['trfQty'] = np.where(roq_df_merged['trfQty'] > roq_df_merged['Batch_Size'], roq_df_merged['trfQty'], roq_df_merged['Batch_Size'])
    roq_df_merged = roq_df_merged.drop(columns=['Batch_Size', 'Warehouse_Loc_ID'])

    optim_df = optim_df[optim_df['From']!='Supplier']
    optim_df = pd.concat([optim_df, roq_df_merged],ignore_index=True)

    # Transform data: prepare to write back into bigquery
    pulp_raw_df = pulp_raw_df.merge(roq_df, left_on=['Warehouse_Loc_ID', 'Product_ID'], right_on=['To', 'Product_ID'], how='left')
    pulp_raw_df['Reorder_Quantity'] = np.where(pulp_raw_df['trfQty'] > pulp_raw_df['Batch_Size'], pulp_raw_df['trfQty'], pulp_raw_df['Batch_Size'])
    pulp_raw_df = pulp_raw_df[pulp_final_column]
    pulp_raw_df['Reorder_Quantity'] = pulp_raw_df['Reorder_Quantity'].astype('int')
    pulp_raw_df[['Warehouse_Location', 'Warehouse_ID']] = pulp_raw_df['Warehouse_Loc_ID'].str.split('-', expand=True)


    logging.info("PuLP run successfully!")