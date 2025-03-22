from fastapi import FastAPI, Body, Depends, Header, status, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from typing import Union
from app.schema_data import HealthCheckResponse, BaseParameter
import requests
import time
import json
import os
from datetime import datetime
import logging
from app.customize_logging.custom_logging import CustomizeLogger
import pandas as pd
from io import BytesIO
import psycopg2
from sqlalchemy import create_engine
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from functools import lru_cache
import pickle

### define config logging format json file
logger = logging.getLogger(__name__)
config_path="app/customize_logging/logging_config.json"

mltoken = {"user": "dev", "key": "1234567890"}


### Run fastapi initialize application 
app = FastAPI(
    title='ML Playgrounds API', 
    description="", 
    version=1
)
logger = CustomizeLogger.make_logger(config_path)
app.logger = logger
app_predict_v1 = APIRouter()

DB_CONFIG = {
    'host': 'host1',
    'database': 'database1',
    'user': 'user1',
    'password': 'password1',
    'port': '5432'
}
engine = create_engine(f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")

def transform_columns_numerical(based_data, based_column, column_encoder):
  with open(f"{column_encoder}.pkl", "rb") as f:
      encoder = pickle.load(f)
      based_data[based_column] = encoder.transform(based_data[[based_column]])

def transform_columns_categorical(based_data, based_column, column_encoder):
  with open(f"{column_encoder}.pkl", "rb") as f:
      encoder = pickle.load(f)
      based_data[based_column] = encoder.transform(based_data[based_column])

def get_data_from_db(query):
    try:
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def load_data_to_db(df, table_name):
    try:
        df.to_sql(table_name, engine, if_exists="append", index=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


### exception handler for page not found - 404
# Custom 404 JSON response
async def custom_404_handler(request: Request, exc: HTTPException):
    return JSONResponse(content={"message": "Page not found"}, status_code=404)

# Catch-all route for 404 handler
@app.exception_handler(404)
async def not_found(request: Request, exc: HTTPException):
    return await custom_404_handler(request, exc)

### For handling general/custom request reponse
class ResponseException(Exception):
    def __init__(self, uuid: str, msg: str, status_code: int):
        self.uuid = uuid
        self.msg = msg
        self.status_code = status_code

### fastApi custom handling with uvicorn response For handling URL is not accessible [504]
@app.exception_handler(ResponseException)
async def response_exception_handler(request: Request, exc: ResponseException):
    response_body_url_err = {
        "id": "{}".format(exc.uuid),
        "timestamp": str(datetime.now()),
        "message": str(exc.msg)
    }
    request.app.logger.info("Get data response_body_url_err : {}".format(response_body_url_err))
    request.app.logger.info("Get data status_code : {}".format(int(exc.status_code)))

    return JSONResponse(
        status_code=int(exc.status_code),
        content=response_body_url_err,
    )


### route handlers
@app.get(
    "/health", 
    tags=["Health Check"], 
    response_model=HealthCheckResponse,
    description="Check health status every endpoint"
    )
async def read_root() -> dict:
    return {
        "status": "UP",
        "checks": [
            {
                "name": "Enpoint /v1/basci-api ",
                "status": "UP"
            }
        ]
    }


@app.post('/api/v1/get_ml_metrics',
                          tags=["api"],
                          description="return parameters"
                    )
async def get_ml_metrics(
        request: Request,
        Oldpeak: str = Form(...),
        Age: str = Form(...),
        Cholesterol: str = Form(...),
        MaxHR: str = Form(...),
        Sex: str = Form(...),
        ChestPainType: str = Form(...),
        ExerciseAngina: str = Form(...),
        ST_Slope: str = Form(...),
        ml_token: Union[str, None] = Header(default=None, convert_underscores=False),
    ):
    if mltoken['key'] == ml_token:
        data_raw = {'Age': [Age],
                    'Sex': [Sex],
                    'ChestPainType': [ChestPainType],
                    'Cholesterol': [Cholesterol],
                    'FastingBS': [0],
                    'MaxHR': [MaxHR],
                    'ExerciseAngina': [ExerciseAngina],
                    'Oldpeak': [Oldpeak],
                    'ST_Slope': [ST_Slope]}
        # data_raw = {'Age': [49],
        #     'Sex': ['F'],
        #     'ChestPainType': ['NAP'],
        #     'Cholesterol': [180],
        #     'FastingBS': [0],
        #     'MaxHR': [156],
        #     'ExerciseAngina': ['N'],
        #     'Oldpeak': [1.0],
        #     'ST_Slope': ['Flat']}
        inference = pd.DataFrame(data_raw)
        # inference = inference.to_dict(orient='records')

        transform_columns_numerical(inference, 'Oldpeak', 'app/models/oldpeak_encoder')
        transform_columns_numerical(inference, 'Age', 'app/models/age_encoder')
        transform_columns_numerical(inference, 'Cholesterol', 'app/models/cholesterol_encoder')
        transform_columns_numerical(inference, 'MaxHR', 'app/models/maxhr_encoder')
        transform_columns_categorical(inference, 'Sex', 'app/models/sex_encoder')
        transform_columns_categorical(inference, 'ChestPainType', 'app/models/chestpaintype_encoder')
        transform_columns_categorical(inference, 'ExerciseAngina', 'app/models/exerciseangina_encoder')
        transform_columns_categorical(inference, 'ST_Slope', 'app/models/st_slope_encoder')

        with open(f"app/models/classifier.pkl", "rb") as f:
            classifier_lr = pickle.load(f)
        prediction = classifier_lr.predict_proba(inference)
        return {f"probability to have heart disease : {prediction[0][1]}"}
        
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Forbidden"
        )