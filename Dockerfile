FROM ubuntu:latest
MAINTAINER Ratan Sebastian "ratan.sebastian@commercetools.de"
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
COPY requirements.txt  /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
COPY app.py /app/app.py
COPY predict_text.py /app/predict_text.py
COPY model.h5 /app/model.h5
COPY model.json /app/model.json
COPY meme/ /app/meme
COPY impact.ttf /app/impact.ttf
COPY CaptionsClean.txt /app/CaptionsClean.txt
ENTRYPOINT ["python"]
CMD ["app.py"]
