"""Student Performance Prediction - Flask application."""

import os

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

import database
import predictor

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

database.init_db()


@app.context_processor
def inject_globals():
    return {"app_name": "EduPredict"}


@app.template_filter("num")
def num(value):
    """Print 45.0 as 45 but leave 62.5 alone."""
    return int(value) if float(value).is_integer() else value


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html", spec=predictor.INPUT_SPEC, form={}, errors={})


@app.route("/predict", methods=["POST"])
def predict():
    values, errors = predictor.validate(request.form)
    if errors:
        return (
            render_template(
                "index.html",
                spec=predictor.INPUT_SPEC,
                form=request.form,
                errors=errors,
            ),
            400,
        )

    # maxlength on the input is client-side only, so cap it here too.
    name = (request.form.get("student_name") or "").strip()[:60] or "Anonymous student"

    try:
        result = predictor.predict(values)
    except predictor.ModelNotTrained as exc:
        return render_template("error.html", message=str(exc)), 503

    result["student_name"] = name
    result["id"] = database.save_prediction(result)
    return render_template("result.html", r=result)


@app.route("/history")
def history():
    return render_template(
        "history.html",
        rows=database.get_predictions(),
        stats=database.get_stats(),
    )


@app.route("/history/<int:pred_id>/delete", methods=["POST"])
def delete(pred_id):
    database.delete_prediction(pred_id)
    flash("Record deleted.", "info")
    return redirect(url_for("history"))


@app.route("/history/clear", methods=["POST"])
def clear():
    database.clear_predictions()
    flash("History cleared.", "info")
    return redirect(url_for("history"))


@app.route("/about")
def about():
    try:
        metrics = predictor.get_metrics()
    except predictor.ModelNotTrained as exc:
        return render_template("error.html", message=str(exc)), 503
    return render_template("about.html", m=metrics, labels=predictor.INPUT_SPEC)


# --------------------------------------------------------------------------- #
# JSON API
# --------------------------------------------------------------------------- #
@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):  # a JSON list or bare value is not usable
        payload = {}
    values, errors = predictor.validate(payload)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400
    try:
        result = predictor.predict(values)
    except predictor.ModelNotTrained as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503
    return jsonify({"ok": True, "result": result})


@app.route("/api/history")
def api_history():
    return jsonify(
        {"stats": database.get_stats(), "rows": database.get_predictions(50)}
    )


@app.errorhandler(404)
def not_found(_e):
    return render_template("error.html", message="Page not found."), 404


if __name__ == "__main__":
    import os
    import threading
    import webbrowser

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(0.8, lambda: webbrowser.open("http://127.0.0.1:5000")).start()

    app.run(debug=True, port=5000)
