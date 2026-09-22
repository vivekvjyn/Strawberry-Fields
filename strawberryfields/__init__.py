from flask import Flask
from flask_cors import CORS

from strawberryfields.config import Config


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    CORS(app)

    from strawberryfields.views import bp
    app.register_blueprint(bp)

    return app
