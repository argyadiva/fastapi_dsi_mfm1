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
        file: UploadFile = File(...), 
        user_name: str = Form(...),
        ml_topic: str = Form(...),
        ml_token: Union[str, None] = Header(default=None, convert_underscores=False),
    ):
    if mltoken['key'] == ml_token:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Invalid file format")
        try:
            # Read CSV
            contents = await file.read()
            csv_data = pd.read_csv(BytesIO(contents))
            
            # Get data from DB (Modify the query accordingly)
            db_data = get_data_from_db("SELECT * FROM table")
            pd.concat([csv_data, db_data],axis=0)
            load_data_to_db(db_data, 'ml_leaderboard')

            return {'project':'project_dummy', 'metrics':'metrics_dummy'}

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Forbidden"
        )