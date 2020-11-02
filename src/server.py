from flask import Flask, request
from jsonschema import validate, ValidationError

from .cli import data_cli

__doc__ = """
Server runs a flask server and use Encoder to
encode the request into tensorflow api. 
"""

app = Flask(__name__)
app.config.from_pyfile("settings.cfg")
app.cli.add_command(data_cli)

request_schema = {
    "type": "array",
    "default": [],
    "items": {"type": "array", "default": [], "items": {"type": "string"}},
}


@app.errorhandler(ValidationError)
def handle_error(err):
    return {"message": str(err)}


@app.route("/v1/prediction", methods=["POST"])
def prediction():
    data = request.get_json()
    validate(data, request_schema)

    # encode
    # request api from tensorflow

    return "Ok"


if __name__ == "__main__":
    app.run()