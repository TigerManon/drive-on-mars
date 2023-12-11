from fastapi import FastAPI
import numpy as np

api = FastAPI()

# define a root `/` endpoint
@api.get("/")
def index():
    return {"ok": "API connected check"}


@api.get("/predict")
def predict(feature1, feature2):

    # model = picle.load_model()
    # prediction = model.predict(feature1, feature2)

    # Here, I'm only returning the features, since I don't actually have a model.
    # In a real life setting, you would return the predictions.

    return {'prediction': int(feature1)*int(feature2)}

@api.get("/mvp")
def mvp(nb1: int, nb2: int, nb3: int, nb4: int):

    arr = np.array([[nb1, nb2], [nb3, nb4]])

    # fct inversement tab (à vérifier)
    flip_flop = np.flip(arr)

    # Convert° en liste de listes
    flipped_list = flip_flop.tolist()

    return {"flipped_array": flipped_list}
