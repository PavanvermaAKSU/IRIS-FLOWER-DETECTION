from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import numpy as np

from database import Base
from database import engine
from database import SessionLocal

from models import Prediction

app = FastAPI()

model = joblib.load("iris_model.pkl")

Base.metadata.create_all(
    bind=engine
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("iris_model.pkl")

class IrisInput(BaseModel):

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

species = [
    "Iris Setosa",
    "Iris Versicolor",
    "Iris Virginica"
]

@app.get("/")
def home():
    return {"message": "Iris Flower Classification API"}

@app.post("/predict")
def predict(data: IrisInput):

    features = np.array([
        [
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]
    ])

    flower_images = {
      "Iris-setosa": "/images/setosa.jpg",
      "Iris-versicolor": "/images/versicolor.jpg",
      "Iris-virginica": "/images/virginica.webp"
    }
    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    confidence = round(
        max(probabilities) * 100,
        2
    )

    db = SessionLocal()

    record = Prediction(
        species=prediction,
        confidence=confidence
    )

    db.add(record)

    db.commit()

    db.close()

    return {
        "prediction": prediction,
        "confidence": confidence,
        "image": flower_images[prediction]

    }


@app.get("/history")
def history():

    db = SessionLocal()

    records = db.query(
        Prediction
    ).order_by(
        Prediction.id.desc()
    ).all()

    result = []

    for row in records:

        result.append({
            "id": row.id,
            "species": row.species,
            "confidence": row.confidence,
            "created_at":
            row.created_at.strftime(
                "%d-%m-%Y %H:%M"
            )
        })

    db.close()

    return result

@app.get("/dashboard")
def dashboard():

    return {

        "accuracy": 98.7,

        "predictions": 125,

        "models": 4,

        "classes": 3,

        "species": {
            "setosa": 50,
            "versicolor": 50,
            "virginica": 50
        },

        "model_comparison": [

            {
                "model": "Random Forest",
                "accuracy": 99
            },

            {
                "model": "SVM",
                "accuracy": 98
            },

            {
                "model": "KNN",
                "accuracy": 97
            },

            {
                "model": "Logistic Regression",
                "accuracy": 96
            }

        ]
    }

@app.get("/model-info")
def model_info():

    return {
        "model_name": "Random Forest",
        "accuracy": "100%",
        "dataset": "Iris.csv",
        "features": 4,
        "classes": 3
    }

@app.get("/last-prediction")
def last_prediction():

    db = SessionLocal()

    row = db.query(
        Prediction
    ).order_by(
        Prediction.id.desc()
    ).first()

    db.close()

    if not row:
        return {}

    return {
        "species": row.species,
        "confidence": row.confidence,
        "created_at":
        row.created_at.strftime(
            "%d %b %Y %I:%M %p"
        )
    }