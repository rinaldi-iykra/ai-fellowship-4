from project.data.query import *
from project.models.forecast_model import *
from project.models.pulp_solver import *

# Step 1: Run the forecast model
forecast_job = ProductForecast()

# Step 2: Query data for dashboard and pulp_solver
forecast_job.collect_data()
forecast_job.preprocess()
forecast_job.predict()
forecast_job.back_transform_data()
forecast_job.write_fcst_bigquery()

# # Step 2: Query data for dashboard and pulp_solver
get_stock_pivot_table()
get_cost_table()
get_pulp_raw()
get_metrics_raw()

pulp_solver()