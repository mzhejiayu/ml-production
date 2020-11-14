FROM python:3.8.5

RUN pip install pipenv

WORKDIR /app

# update  Pipfile
ADD Pipfile .

# Install the dependences
RUN pipenv install --skip-lock

ADD src /app/src
ADD pipeline /pipeline
WORKDIR /app/src

CMD [ "pipenv", "run", "flask", "run" ]