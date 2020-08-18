FROM ubuntu:18.04

MAINTAINER vamsi

RUN apt-get update && apt -y install python3-pip && apt-get install -y libsm6 libxext6 libxrender-dev

COPY ./requirements.txt /vamsi1/requirements.txt

WORKDIR /vamsi1

RUN pip3 install -r requirements.txt

COPY . /vamsi1

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]
