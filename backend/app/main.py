from flask import Flask, jsonify
from flask import Flask, jsonify, render_template

from app.api import (
    behavioural_bp,
    facial_bp,
    language_bp,
    physiological_bp,
    scores_bp,
)


def create_app() -> Flask:
    app = Flask(__name__)
    @app.get("/")
    def language_analysis_portal():
        return render_template("language_analysis.html")

    app.register_blueprint(scores_bp)
    app.register_blueprint(language_bp)
    app.register_blueprint(behavioural_bp)
    app.register_blueprint(physiological_bp)
    app.register_blueprint(facial_bp)

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "service": "j26-it-402-backend",
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )