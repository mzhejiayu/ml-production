import json
import os
import requests
from flask import Flask, current_app, g, has_app_context, jsonify, request, abort
from joblib import load
from jsonschema import ValidationError, validate

from .encoder import Encoder

from .cli import data_cli

__doc__ = """
Server runs a flask server and use Encoder to
encode the request into tensorflow api. 
"""

app = Flask(__name__)
app.config.from_pyfile("settings.cfg")

# Update the configuration with environments starting with MLP_
for key in os.environ.keys():
    if key.startswith("MLP_"):
        env_name = key.split("MLP_")[1]
        app.config[env_name] = os.environ.get(key)


app.cli.add_command(data_cli)

request_schema = {
    "type": "array",
    "default": [],
    "items": {"type": "array", "default": [], "items": {"type": "string"}},
}


def get_encoder():
    if has_app_context():
        path = current_app.config["PIPE_PATH"]
        if "encoder" not in g:
            pipe = load(path)
            g.encoder = Encoder(pipe)
        return g.encoder


@app.errorhandler(ValidationError)
def handle_error(err):
    return {"message": str(err)}


@app.route("/v1/prediction", methods=["POST"])
def prediction():
    data = request.get_json()
    validate(data, request_schema)

    matrix = get_encoder().encode(data).tolist()

    res = requests.post(
        current_app.config["TFS_URL"] + "/v1/models/tp_pred:predict",
        data=json.dumps({"instances": matrix}),
    )

    if res.status_code != 200:
        abort(res.status_code, description=f"{res}")
    else:
        return jsonify(res.json())


def tfs_healthcheck():
    res = requests.get(app.config["TFS_URL"], timeout=0.01)


tfs_healthcheck()