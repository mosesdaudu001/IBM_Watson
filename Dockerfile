FROM python:3.9

ENV PYTHONUNBUFFERED True

RUN apt-get upgrade && apt-get update
RUN pip install -U pip
RUN apt-get install ffmpeg -y


COPY . .

ENV PORT 8000

RUN pip install -r requirements.txt


# Uncomment this section if deploying to docker

# EXPOSE 2000

# # ENTRYPOINT [ "uvicorn", "--timeout=600", "--bind=0.0.0.0:2000", "app:app" ]
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "2000"]


# This section is for deploying to cloud build and cloud run
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1

