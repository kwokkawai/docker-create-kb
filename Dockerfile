FROM python:3.10-slim

# set the working directory
WORKDIR /code

# install dependencies
COPY ./requirements.txt ./
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# copy the src to the folder
COPY ./src ./src
COPY ./data ./data

# start the server
CMD ["streamlit", "run", "src.app:app"]

#streamlit run streamlit_app.py --server.port 8080