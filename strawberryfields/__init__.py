from flask import Flask
from flask_cors import CORS

from strawberryfields.config import Config, load_pitch_config
from strawberryfields import db


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config["PITCH"] = load_pitch_config()

    CORS(app)
    app.config["TRACKS"] = db.load_tracks(app)

    from strawberryfields.views import bp
    app.register_blueprint(bp)

    return app
