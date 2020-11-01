from models import Model
from fastapi import FastAPI, Response
from fastapi import responses
from models import Model
from customtypes import DataItemType, PredItemType

app = FastAPI()
model = Model()


@app.get("/", status_code=200)
async def root(response: Response):
    try:
        return {"message": "HealthCheck Successful"}
    except Exception:
        response.status_code = 400
        return {"message": "Healthcheck failed, server is down"}


@app.post("/predict", status_code=200)
async def predict(data: DataItemType, name: str, response: Response):
    try:
        model.select(name)
        prediction: PredItemType = model.prediction(data)
        return {"message": f"Predicted class is : {prediction}"}
    except IndexError:
        response.status_code = 400
        return {"message": "Specified Model is not available, pick correct model name"}
    except Exception:
        response.status_code = 500
        return {"message": "Something went wrong with the prediction, please try again"}
