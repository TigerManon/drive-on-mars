import streamlit as st
from PIL import Image

import requests

# ----------------------------------------------------------------
# Using local API
url = 'http://127.0.0.1:8000'

# Using our remote running API
# url = 'https://containermars-jnjlxgdooq-ew.a.run.app/predict'



# ----------------------------------------------------------------
# App title and description

st.header('Simple API Call and Image Uploader 📸')
st.markdown('''
            > This is a Le Wagon boilerplate for any data science projects that involve exchanging images between a Python API and a simple web frontend.

            > **What's here:**

            > * [Streamlit](https://docs.streamlit.io/) on the frontend
            > * [FastAPI](https://fastapi.tiangolo.com/) on the backend
            > * [PIL/pillow](https://pillow.readthedocs.io/en/stable/) and [opencv-python](https://github.com/opencv/opencv-python) for working with images
            > * Backend and frontend can be deployed with Docker
            ''')

st.markdown("---")




### Create a native Streamlit file upload input
st.markdown("### Let's do a simple image upload 👇")
img_file_buffer = st.file_uploader('Upload an image')


if img_file_buffer is not None:

  col1, col2 = st.columns(2)

  with col1:
    ### Display the image user uploaded
    st.image(Image.open(img_file_buffer), caption="Here's the image you uploaded ☝️")

  with col2:
    with st.spinner("Wait for it..."):
      ### Get bytes from the file buffer
      img_bytes = img_file_buffer.getvalue()

      ### Make request to  API (stream=True to stream response as bytes)
      res = requests.post(url + "/upload_image", files={'img': img_bytes})

      if res.status_code == 200:
        ### Display the image returned by the API
        st.image(res.content, caption="Image returned from API ☝️")
      else:
        st.markdown("**Oops**, something went wrong 😓 Please try again.")
        print(res.status_code, res.content)



st.markdown("---"*10)

st.markdown('''
            > Additional information

            ''')

param1 = st.slider('Select a number', 1, 10, 3)

param2 = st.slider('Select another number', 1, 10, 3)


params = {
    'feature1': param1,  # 0 for Sunday, 1 for Monday, ...
    'feature2': param2
}
response = requests.get(url, params=params)

# st.text(response.json())
