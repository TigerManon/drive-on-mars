# Instead of creating an image from scratch, we use this image which has python installed.
FROM python:3.10-bookworm

# COPY allows you to select the folders and files to include in your docker image
# Here, we will include our api_folder and the requiremenets.txt file
COPY requirements.txt requirements.txt

# RUN allows you to run terminal commands when your image gets created
# Here, we upgrade pip and install the libraries in our requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY drive_on_mars drive_on_mars
RUN mkdir -p raw_data/models
# CMD controls the functionality of the container
# Here, we use uvicorn to control the web server ports

# deploy
CMD uvicorn drive_on_mars.api.api_file:api --host 0.0.0.0 --port $PORT
