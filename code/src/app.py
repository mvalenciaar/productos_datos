from flask import Flask, render_template

import analytic_model_fraud

app = Flask(__name__)
app.config["SECRET_KEY"] = "you-will-never-guess"


# @app.route("/", methods=["GET", "POST"])
# @app.route("/index", methods=("GET", "POST"))


@app.route("/", methods=["GET", "POST"])
def index():
    analytic_model_fraud.run()
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5006, host="0.0.0.0")