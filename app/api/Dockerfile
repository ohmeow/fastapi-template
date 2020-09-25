FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

ENV MODULE_NAME=app

RUN apt-get update && apt-get install -y ncdu \
    && rm -rf /var/lib/apt/lists/*

COPY ./app/api/requirements.txt /app
RUN pip install --upgrade pip
RUN pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --upgrade -r requirements.txt
RUN rm -rf /root/.cache

COPY ./app/api /app
COPY ./ml-models /ml-models