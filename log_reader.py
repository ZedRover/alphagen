from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf

app = Flask(__name__)
LOG_FOLDER = "log"


def add_empty_lines(log_content: str) -> str:
    lines = log_content.split("\n")
    new_lines = []
    for i, line in enumerate(lines):
        new_lines.append(line)
        if i < len(lines) - 1 and lines[i + 1].startswith("pool/best_ic_ret"):
            new_lines.append("")
    return "\n".join(new_lines)


def get_all_log_files(folder):
    log_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if "events" in file:
                log_files.append(os.path.relpath(os.path.join(root, file), LOG_FOLDER))
    return log_files


@app.route("/")
def index():
    files = get_all_log_files(LOG_FOLDER)
    return render_template("index.html", files=files)


@app.route("/display_log/<path:filename>")
def display_log(filename):
    filepath = os.path.join(LOG_FOLDER, filename)
    logs = []
    for event in tf.compat.v1.train.summary_iterator(filepath):
        for value in event.summary.value:
            if value.HasField("simple_value"):
                if value.tag == "pool/best_ic_ret":
                    logs.append("\n")
                logs.append(f"{value.tag}: {value.simple_value}")
    return render_template("logs.html", logs=logs)


if __name__ == "__main__":
    app.run(debug=True)
