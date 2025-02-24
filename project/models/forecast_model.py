import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from datetime import datetime

from mlforecast import MLForecast
# import mlforecast.flavor
from mlforecast.lag_transforms import RollingMean

from lightgbm import LGBMRegressor

from utilsforecast.evaluation import evaluate
from utilsforecast.losses import rmse

from google.cloud import bigquery
import json

from time import time
import numpy as np
from holidays import ID

import optuna

import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import subprocess
import socket
import os
import signal

import json

import mmap
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import logging

class ProductForecast:
    """
    A class that encompasses the end-to-end process of collecting data, preprocessing, 
    training machine learning models, and making predictions for product demand forecasting.

    Attributes:
        df: pd.DataFrame
            The raw data collected from the data source (e.g., BigQuery).
        prep_df: pd.DataFrame
            The preprocessed DataFrame, ready for training and prediction.
        unique_id_map: dict
            A mapping of unique IDs for different products across various warehouses.
        train: pd.DataFrame
            The training dataset used to train the machine learning model.
        test: pd.DataFrame
            The testing dataset used to evaluate the machine learning model.
        fcst_model: list
            List of forecasting models used, e.g., ['LGBMRegressor'].
        lags: int
            The number of lag features to generate during preprocessing (default is 8).
        horizon: int
            The forecast horizon, i.e., how many days into the future to predict (default is 1).
        eval_metric: dict
            A dictionary of evaluation metrics (e.g., RMSE) for the model's performance.
        y_pred: dict
            The predicted values for the forecasted demand.
        prediction_df: dict
            The DataFrame containing the test data along with the forecasted values.
        port: int
            The port used for the MLflow tracking server (default is 5002).
        tracking_uri: str
            The URI for the MLflow tracking server.
        experiment_name: str
            The name of the MLflow experiment.
        latest_model_uri: dict
            The URI of the most recently trained model stored in MLflow.
        logger: logging.Logger
            Logger instance to log runtime information and errors.
        
    Methods:
        __init__():
            Initializes the ProductForecast class, setting up necessary configurations, 
            paths, and default values.

        close_port(port=5002):
            Closes the process using the specified port to ensure no conflicts when starting 
            the MLflow server.

        is_port_in_use():
            Checks if the specified port is currently in use by any process.

        start_mlflow_server():
            Starts the MLflow tracking server for model training and artifact storage, 
            ensuring the specified port is available.

        get_latest_date():
            Retrieves the most recent date from the data source to ensure the latest data 
            is used for forecasting.

        collect_data():
            Collects data from the BigQuery source, starting from the latest date available 
            in the database and pulling one year of historical data.

        preprocess():
            Preprocesses the raw data by creating necessary features (e.g., holiday features, 
            lag features) and splits the data into training and testing datasets.

        fit_predict():
            Trains the machine learning model using the training data, tunes hyperparameters 
            using Optuna, and evaluates the model on the test data.

        predict():
            Loads the best-trained model from MLflow and uses it to make predictions on the 
            test dataset.

        back_transform_data():
            Transforms the predicted data back to the original format (e.g., mapping unique 
            IDs back to product details, renaming columns) before saving it to the database.

        write_fcst_bigquery():
            Writes the forecasted data back to BigQuery, updating the relevant table with 
            forecasted demand.
    """

    def __init__(self):
        self.bqclient = bigquery.Client()
        self.tableid = 'b-508911.test_dataset_iykra.test_table'
        self.stagingtableid = 'b-508911.test_dataset_iykra.staging_table'     
        self.gsname = 'gs://bucket-capstone-iykra0'
        self.latest_date = None
        self.df = None
        self.prep_df = None
        self.unique_id_map = None
        self.train = None
        self.test = None
        self.fcst_model = ['LGBMRegressor']
        self.lags = 8
        self.horizon = 2
        self.eval_metric = {}
        self.y_pred = {}
        self.prediction_df = {}
        self.port='5003'
        self.tracking_uri = f'http://127.0.0.1:{self.port}'
        self.experiment_name = 'product_forecast'
        self.latest_model_uri = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def close_port(self):
        """
        Closes the process using a specific port by terminating it.

        Args:
            port (int): The port number to close. Default is 5002.

        Raises:
            Exception: If the process using the port cannot be terminated.
        """

        try:
            # Find the process using the port
            result = subprocess.run(["lsof", "-i", f":{self.port}"], capture_output=True, text=True)
            lines = result.stdout.splitlines()
            
            if len(lines) > 1:
                # Extract the PID from the second line
                pid = int(lines[1].split()[1])
                print(f"Terminating process {pid} using port {self.port}")
                
                # Terminate the process
                os.kill(pid, signal.SIGTERM)
                print(f"Process {pid} terminated")
            else:
                print(f"No process found using port {self.port}")
        except Exception as e:
            print(f"Failed to close port {self.port}: {e}")

    def is_port_in_use(self):
        """
        Checks whether a specific port is currently in use.

        Returns:
            bool: True if the port is in use, False otherwise.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', int(self.port))) == 0
        
    def start_mlflow_server(self):
        """
        Start the MLflow tracking server with local backend store and GCS artifact storage.
        """
        if self.is_port_in_use():
            print(f"Port {self.port} is already in use. Closing port.")
            self.close_port()
        try:
            # Start the MLflow tracking server
            subprocess.Popen([
                "mlflow", "server",
                "--backend-store-uri", os.path.expanduser('project/models/mlruns/'),
                "--default-artifact-root", f"{self.gsname}/mlruns",
                "--host", "127.0.0.1",
                "--port", str(self.port)
            ])
            print(f"MLflow server is starting on port {self.port}")
        except Exception as e:
            print(f"Failed to start MLflow server: {e}")

    def get_latest_date(self):
        """
        Fetches the latest date from the BigQuery table to limit the data collection range.

        Raises:
            Exception: If the query to fetch the latest date fails.
        """
        query_latest_date = f"""
        select max(Date) as latest_date from {self.tableid}
        """
        self.latest_date = self.bqclient.query(query_latest_date).to_dataframe()['latest_date'].iloc[0]
     
    def collect_data(self):
        """
        Collects the data from a BigQuery table for the past year and stores it in `self.df`.
        """
        self.logger.info('Collecting data...')
        self.get_latest_date()

        query_all_test_table = f"""
        select Date, Warehouse_ID, Warehouse_Location, Product_ID, Discount_Impact, Sales_Event, Daily_Sales
        from {self.tableid}
        where Date >= date_sub('{str(self.latest_date)}', interval 1 year)
        """

        self.df = self.bqclient.query(query_all_test_table).to_dataframe()
        self.logger.info('Collecting data: complete!')

    def preprocess(self):
        """
        Preprocesses the collected data by:
        1. Creating unique IDs.
        2. Processing holiday features.
        3. Generating lag features.
        4. Splitting the data into training and testing sets.

        Logs resource usage before and after preprocessing.
        """
        self.logger.info('Preprocess data...')
        # Convert Date to datetime and create unique_id in one pass
        data = self.df.assign(
            ds=lambda x: pd.to_datetime(x['Date']),
            y=self.df['Daily_Sales'],
            unique_id=lambda x: x['Warehouse_Location'] + '-' + x['Warehouse_ID'] + '_' + x['Product_ID']
        )
        def process_unique_ids(data):
            file_path = 'project/models/unique_id_map.json'
            logging.info(f'File path to unique_id_map: {file_path}')
            # Memory-mapped file reading
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    try:
                        self.unique_id_map = json.loads(mm.read().decode())
                        data['unique_id'] = data['unique_id'].map(self.unique_id_map)
                    finally:
                        mm.close()
            else:
                # Process in chunks for large datasets
                chunk_size = 1000
                unique_ids = {}
                
                for chunk in data['unique_id'].groupby(np.arange(len(data)) // chunk_size):
                    unique_ids.update(dict.fromkeys(chunk[1].unique()))
                
                self.unique_id_map = {k: i for i, k in enumerate(unique_ids)}
                data['unique_id'] = data['unique_id'].map(self.unique_id_map)
                
                # Async file writing
                with ThreadPoolExecutor() as executor:
                    executor.submit(lambda: json.dump(self.unique_id_map, open(file_path, 'w')))
            
            return data
        
        data = process_unique_ids(data)

        # Process Indonesian holidays
        current_year = data['Date'].max().year
        indo_holiday = ID(years=[current_year - 1, current_year, current_year + 1])

        # Categorize holidays
        gen_holiday = [datetime.strptime(str(i), '%Y-%m-%d') 
                       for i in indo_holiday if indo_holiday[i] not in \
                       ['New Year\'s Day', 'Christmas Day', 'Eid al-Fitr', 'Eid al-Fitr Second Day']]
        eid_fitr = [datetime.strptime(str(i), '%Y-%m-%d') 
                    for i in indo_holiday if indo_holiday[i] in ['Eid al-Fitr', 'Eid al-Fitr Second Day']]
        xmas_newyear = [datetime.strptime(str(i), '%Y-%m-%d') 
                        for i in indo_holiday if indo_holiday[i] in ['Christmas Day', 'New Year\'s Day']]

        data = data[['ds', 'unique_id', 'Discount_Impact', 'Sales_Event', 'y']]

        # Function to calculate days until the next holiday
        def days_until_next_holiday(date, holiday_list):
            future_dates = [h for h in holiday_list if h > date]
            if future_dates:
                return (min(future_dates) - date).days
            return 365

        # Calculate holiday differences (vectorized with delayed)
        data['gen_holiday_dayleft'] = data['ds'].apply(lambda x: days_until_next_holiday(x, gen_holiday))
        data['eid_fitr_dayleft'] = data['ds'].apply(lambda x: days_until_next_holiday(x, eid_fitr))
        data['xmas_newyear_dayleft'] = data['ds'].apply(lambda x: days_until_next_holiday(x, xmas_newyear))
        data = data.sort_values(by=['unique_id', 'ds'])

        # Keep only necessary columns and sort
        data = data[['ds', 'unique_id', 'Discount_Impact',
                     'gen_holiday_dayleft', 'eid_fitr_dayleft',
                     'xmas_newyear_dayleft', 'Sales_Event', 'y']].sort_values(['unique_id', 'ds'])

        # Configure and run MLForecast preprocessing
        preprocess_model = MLForecast(
            models=[],
            freq='D',
            lags=range(1, self.lags),
            lag_transforms={1: [RollingMean(window_size=7, min_samples=1)]},
            date_features=['dayofweek', 'week', 'month', 'year'],
            num_threads=2
        )
        
        self.prep_df = preprocess_model.preprocess(data, static_features=[], dropna=True)

        # Split train/test using iloc for better performance
        train_date = self.prep_df['ds'].max() - pd.Timedelta(days=self.horizon)

        # Use boolean indexing directly instead of isin()
        self.train = self.prep_df[self.prep_df['ds'] <= train_date]
        self.test = self.prep_df[self.prep_df['ds'] > train_date]
        self.logger.info('Preprocess data: complete!')

    def fit_predict(self):
        """
        Fits and tunes machine learning models (e.g., LGBMRegressor) on the training data, evaluates them, 
        and makes predictions on the test data.

        Logs the best hyperparameters and model performance metrics (e.g., RMSE).
        """
        self.logger.info('Fit and predict data...')
        # self.start_mlflow_server()
        # mlflow.set_tracking_uri(uri=self.tracking_uri)
        # mlflow.set_experiment(self.experiment_name)
        optuna.logging.set_verbosity(optuna.logging.INFO)

        self.fcst_model = ['LGBMRegressor']
        
        # with mlflow.start_run():
            # -------Train and log LGBMRegressor-----------
            # train_ds = mlflow.data.from_pandas(self.train)
            # test_ds = mlflow.data.from_pandas(self.test)            
            # mlflow.log_input(train_ds, context="training")
            # mlflow.log_input(test_ds, context="test")
            
        def objective(trial):
            lgbr_params = {
                'verbose':-1,
                'force_col_wise':True,
                'random_state':21,
                'n_estimators': 500,
                'learning_rate' : 0.1,
                'num_leaves' : trial.suggest_int('num_leaves', 2, 256),
                'min_data_in_leaf' : trial.suggest_int('min_data_in_leaf', 1, 100),
                'bagging_fraction'  : trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
                'colsample_bytree' : trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
            }
            params = {
                'init': {
                    'models': {
                        self.fcst_model[0]: LGBMRegressor(**lgbr_params),
                    },
                    'freq': 'D',
                    'num_threads': 2,
                },
                'fit': {
                    'static_features': ['unique_id'],
                }
            }

            ml_fcst = MLForecast(**params['init'])
        
            ml_fcst.fit(self.train, **params['fit'])

            self.y_pred[self.fcst_model[0]] = ml_fcst.predict(h=self.horizon, X_df=self.test)
            self.prediction_df[self.fcst_model[0]] = self.test.merge(
                self.y_pred[self.fcst_model[0]],
                on=['unique_id','ds'], how='left'
            )

            error = evaluate(
                self.prediction_df[self.fcst_model[0]],
                metrics=[rmse],
                agg_fn='mean',
            )[self.fcst_model[0]].values[0]

            return error
        
        self.logger.info('Optuna trial starts...')
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=5)
        self.logger.info('Optuna trial finished!')

        # Log the best hyperparameters and metrics
        # signature = infer_signature(self.train.head(1), self.y_pred[self.fcst_model[0]])
        # mlflow.log_params(study.best_params)
        # mlflow.log_metrics({'rmse': study.best_trial.value})
        self.eval_metric[f'{self.fcst_model[0]}_rmse'] = study.best_trial.value

        # Retrain the model with the best hyperparameters
        lgbr_params = {
            'verbose': -1,
            'force_col_wise': True,
            'random_state': 21,
            'n_estimators': 500,
            'learning_rate': 0.1,
            'num_leaves': study.best_params['num_leaves'],
            'min_data_in_leaf': study.best_params['min_data_in_leaf'],
            'bagging_fraction': study.best_params['bagging_fraction'],
            'colsample_bytree': study.best_params['colsample_bytree'],
        }

        params = {
            'init': {
                'models': {
                    self.fcst_model[0]: LGBMRegressor(**lgbr_params),
                },
                'freq': 'D',
                'num_threads': 2,
            },
            'fit': {
                'static_features': ['unique_id'],
            }
        }
        
        # Train the final model with the best parameters
        ml_fcst = MLForecast(**params['init'])
        ml_fcst.fit(self.train, **params['fit'])
        ml_fcst.save('project/models/lgbmregressor.pkl')

        # Log the model to MLflow
        # artifact_path = 'model'
        # mlforecast.flavor.log_model(model=ml_fcst, artifact_path=artifact_path, signature=signature)
        # self.latest_model_uri[self.fcst_model[0]] = mlflow.get_artifact_uri(artifact_path)
        # mlflow.register_model(self.latest_model_uri[self.fcst_model[0]],self.fcst_model[0])
        self.logger.info('Fit and predict: complete!')

    def predict(self):
        """
        Generates predictions using the latest trained model and stores the results in `self.prediction_df`.
        """
        self.logger.info('Predict data...')
        # self.start_mlflow_server()
        # mlflow.set_tracking_uri(uri=self.tracking_uri)
        # mlflow.set_experiment(self.experiment_name)
        # client = MlflowClient()
        # self.fcst_model = ['LGBMRegressor']
        # latest_version = client.get_latest_versions(name=self.fcst_model[0])[0].source
        # loaded_model = mlforecast.flavor.load_model(model_uri=latest_version)
        loaded_model = MLForecast.load('project/models/lgbmregressor.pkl')
        self.y_pred[self.fcst_model[0]] = loaded_model.predict(X_df = self.test, h=self.horizon)
        self.prediction_df[self.fcst_model[0]] = self.test.merge(
            self.y_pred[self.fcst_model[0]],
            on=['unique_id','ds'], how='left'
        )
        self.logger.info('Predict data complete!')

    def back_transform_data(self):
        """
        Writes the forecasted data to BigQuery, updating the target table with forecasted demand.

        Logs resource usage before and after writing the forecast data.

        Raises:
            Exception: If the write or update process fails.
        """
        self.logger.info('Back transform data...')

        # Transform dataframe back to original
        prediction_df = self.prediction_df[self.fcst_model[0]]
        column_to_keep= ['ds','y','Discount_Impact', 'Sales_Event', 'LGBMRegressor', 'Warehouse_ID', 'Warehouse_Location', 'Product_ID']
        prediction_df['LGBMRegressor'] = prediction_df['LGBMRegressor'].astype('int')
        prediction_df['unique_id'] = prediction_df['unique_id'].map({v: k for k, v in self.unique_id_map.items()})
        prediction_df[['Warehouse_Loc_ID', 'Product_ID']] = prediction_df['unique_id'].str.split('_', expand=True)  
        prediction_df[['Warehouse_Location', 'Warehouse_ID']] = prediction_df['Warehouse_Loc_ID'].str.split('-', expand=True)      
        prediction_df = prediction_df[column_to_keep]
        prediction_df.rename(columns={'ds':'Date', 'y':'Daily_Sales','LGBMRegressor':'Forecasted_Demand'}, inplace=True)
        self.prediction_df[self.fcst_model[0]] = prediction_df
        self.logger.info('Back transform data: complete!')

    def write_fcst_bigquery(self):
        """
        Writes the forecasted data to BigQuery, updating the target table with forecasted demand.

        Raises:
            Exception: If the write or update process fails.
        """
        self.logger.info('Writing into bigquery...')
        self.bqclient.delete_table(self.stagingtableid, not_found_ok=True)
        
        if not self.latest_date:
            query_latest_date = f"""
            SELECT MAX(Date) AS latest_date
            FROM `{self.tableid}`
            """
            self.latest_date = self.bqclient.query(query_latest_date).to_dataframe()['latest_date'].iloc[0]

        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
        )
        load_job = self.bqclient.load_table_from_dataframe(self.prediction_df[self.fcst_model[0]], self.stagingtableid, job_config=job_config)
        load_job.result()

        update_query = f"""
        MERGE `{self.tableid}` AS target
        USING `{self.stagingtableid}` AS source
        ON target.Product_ID = source.Product_ID 
            AND target.Warehouse_ID = source.Warehouse_ID
            AND target.Warehouse_Location = source.Warehouse_Location
            AND target.Date = source.Date
        WHEN MATCHED THEN
        UPDATE SET target.Forecasted_Demand = source.Forecasted_Demand
        """

        update_job = self.bqclient.query(update_query)
        update_job.result()

        self.logger.info(f"Rows with the latest date in `{self.tableid}` successfully updated.")
