import json

from fastapi import FastAPI, Response, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd

from models import Model
from customtypes import DataItemType


app = FastAPI()
model = Model()

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", status_code=200)
async def root(response: Response):
    try:
        return {"message": "HealthCheck Successful"}
    except Exception:
        response.status_code = 400
        return {"message": "Healthcheck failed, server is down"}


@app.post("/predict", status_code=200)
async def predict(data: DataItemType, response: Response):
    try:
        model.select(data.name)
        prediction: str = model.prediction(data)
        return {"message": f"Predicted class is : {prediction}"}
    except IndexError:
        response.status_code = 400
        return {"message": "Specified Model is not available, pick correct model name"}
    except Exception as e:
        response.status_code = 500
        print(str(e))
        return {"message": "Something went wrong with the prediction, please try again"}


@app.post("/test", status_code=200)
async def test(file: UploadFile = File(...)):
    df = ConvertBytesToDataFrame(file)
    return model.test


def ConvertBytesToDataFrame(bytes):
    data = bytes.decode('utf-8').splitlines()
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
