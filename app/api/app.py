import os
import pickle
import numpy as np

from fastapi import FastAPI 
from aiohttp import ClientSession
from schemas import ( RequestBody, ResponseBody, LabelResponseBody, ResponseValues, TextSample )
from utils import wait_for_debugger

# for vscode debugging
if (os.getenv("WHEREAMI") == 'development'): wait_for_debugger()

# build app object and load our model on startup
app = FastAPI(title="simple-model", description="a simple model-serving skateboard in FastAPI", version="0.1")
with open(os.getenv("MODEL_PATH"), "rb") as rf: 
    clf = pickle.load(rf)

# build our aiohttp.ClientSession object for managing any async requests
client_session = ClientSession()

# routes
@app.get("/healthcheck")
async def healthcheck():
    msg = (
        "this sentence is already halfway over, "
        "and still hasn't said anything at all very good :) maybedddd"
    )
    return {"message": msg}


@app.post("/predict", response_model=ResponseBody)
async def predict(body: RequestBody):
    data = np.array(body.to_array())

    probas = clf.predict_proba(data)
    predictions = probas.argmax(axis=1)

    return {
        "predictions": (
            np.tile(clf.classes_, (len(predictions), 1))[np.arange(len(predictions)), predictions].tolist()
        ),
        "probabilities": probas[np.arange(len(predictions)), predictions].tolist()
    }


@app.post("/predict/{label}", response_model=LabelResponseBody)
async def predict_label(label: ResponseValues, body: RequestBody):
    data = np.array(body.to_array())

    probas = clf.predict_proba(data)
    target_idx = clf.classes_.tolist().index(label.value)

    return {"label": label.value, "probabilities": probas[:, target_idx].tolist()}


@app.get("/cat-facts", response_model=TextSample)
async def cat_facts():
    url = "https://cat-fact.herokuapp.com/facts/random"
    async with client_session.get(url) as resp:
        response = await resp.json()

    return response


# lifecycle methods
@app.on_event("shutdown")
async def cleanup():
    await client_session.close()