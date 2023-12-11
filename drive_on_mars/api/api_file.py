from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

import numpy as np
import cv2
import io

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


@api.post('/upload_image')
async def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()

    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray
    print('*'*10,'\n', type(cv2_img),'\n','*'*10)
    ### Do cool stuff with your image.... For example face detection
    annotated_img = cv2_img

    ### Encoding and responding with the image
    im = cv2.imencode('.png', annotated_img)[1] # extension depends on which format is sent from Streamlit
    return Response(content=im.tobytes(), media_type="image/png")
