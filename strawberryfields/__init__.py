import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS


def create_app():
    load_dotenv()

    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = (f"{database_url}{separator}"
                        f"sslmode={os.environ.get('DB_SSLMODE', 'require')}")
    else:
        database_url = (f"postgresql://{os.environ.get('DB_USER')}:"
                        f"{os.environ.get('DB_PASSWORD')}@"
                        f"{os.environ.get('DB_HOST', 'localhost')}:"
                        f"{os.environ.get('DB_PORT', '5432')}/"
                        f"{os.environ.get('DB_NAME', 'strawberry_fields')}")

    with (Path(__file__).resolve().parent.parent / "config.yaml").open(
        encoding="utf-8"
    ) as handle:
        pitch = yaml.safe_load(handle)["app"]

    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")
    app.config["DATABASE_URL"] = database_url
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.config["PITCH"] = pitch

    CORS(app)

    from strawberryfields.views import bp
    app.register_blueprint(bp)

    return app
