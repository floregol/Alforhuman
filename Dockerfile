FROM ubuntu:22.04
EXPOSE 5000
VOLUME /usr/src/app/
WORKDIR /usr/src/app/
RUN apt update && apt install -y python3-pip build-essential
COPY requirements.txt .
COPY src/generateColor.py .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN python3 generateColor.py
ARG FLASK_APP=server.py
CMD ["flask", "run", "--host=0.0.0.0"]
