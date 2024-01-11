from fastapi import FastAPI, UploadFile, File
from starlette.responses import Response

import numpy as np
import cv2

from drive_on_mars.model.registry import load_model

api = FastAPI()
api.state.model = load_model()

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
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    ### Pre-processing: resizing
    img_preproc = cv2.resize(cv2_img, dsize = (256, 256))
    ######### >>>>>>>>>> Check how resizing is done in the model
    img_preproc_arr = np.array([img_preproc])
    print("="*30,'>>> preproc image now shape:',img_preproc_arr.shape)

    ### Loading
    model = api.state.model
    if model is None:
        print('-'*30,"MODEL IS NONE")

    ### Model prediction for each class
    y_pred_probas_arr = model.predict(img_preproc_arr)
    print('Does the predict work? shape:',y_pred_probas_arr.shape)

    ### Prediction image
    y_pred_arr = np.argmax(y_pred_probas_arr[:,:,:,:5], axis=3)
    y_pred = y_pred_arr[0].astype(np.uint8)
    print('Does the predicted image work? shape:',y_pred.shape)

    return Response(content=y_pred.tobytes(), media_type="image/png")

    # output_image = cv2.resize(y_pred, dsize = (1024, 1024))

    ### Encoding and responding with the image
    # im = cv2.imencode('.png', output_image)[1] # extension depends on which format is sent from Streamlit

    # return Response(content=im.tobytes(), media_type="image/png")
