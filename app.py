# app.py (NO PUMP VERSION)
import os
from datetime import datetime
from collections import deque
from flask import Flask, request, jsonify
import joblib
import numpy as np
import threading
import csv

import mysql.connector
from mysql.connector import Error

from flask_cors import CORS

# ========== CONFIG ==========
MODEL_FILE = os.environ.get("MODEL_FILE", "model_random_forest.pkl")
ENCODER_FILE = os.environ.get("ENCODER_FILE", "label_encoder.pkl")
USE_MYSQL = os.environ.get("USE_MYSQL", "false").lower() == "true"

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_USER = os.environ.get("DB_USER", "root")
DB_PASS = os.environ.get("DB_PASS", "")
DB_NAME = os.environ.get("DB_NAME", "db_kebakaran")

LOG_CSV = os.environ.get("LOG_CSV", "logs/fire_data.csv")
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "240"))

# =====================================

app = Flask(__name__)
CORS(app, origins="*")

model = joblib.load(MODEL_FILE)
label_encoder = joblib.load(ENCODER_FILE)

history = deque(maxlen=MAX_HISTORY)

# Only BUZZER mode retained
buzzer_state = {"mode": "OFF"}  # OFF / WARN / DANGER

# Create logs folder
os.makedirs(os.path.dirname(LOG_CSV) or ".", exist_ok=True)
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "temp", "hum", "gas", "flame", "status"])

# Optional DB
db_conn = None
db_cursor = None

def connect_db():
    global db_conn, db_cursor
    if not USE_MYSQL:
        return False
    try:
        db_conn = mysql.connector.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASS,
            database=DB_NAME, autocommit=False
        )
        db_cursor = db_conn.cursor()
        print("[MYSQL] Connected")
        return True
    except Error as e:
        print("DB Error:", e)
        return False

def ensure_db_connection():
    if not USE_MYSQL:
        return False
    try:
        if db_conn is None or not db_conn.is_connected():
            return connect_db()
        return True
    except:
        return connect_db()

def insert_row(temp, hum, gas, flame, status):
    if not USE_MYSQL:
        return False
    if not ensure_db_connection():
        return False
    try:
        sql = ("INSERT INTO sensor_data "
               "(timestamp, temperature, humidity, gas, flame, status) "
               "VALUES (%s, %s, %s, %s, %s, %s)")
        tstamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        db_cursor.execute(sql, (tstamp, temp, hum, gas, flame, status))
        db_conn.commit()
        return True
    except Exception as e:
        print("Insert Err:", e)
        return False

# CSV logging
def append_csv(entry):
    try:
        with open(LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([entry["timestamp"], entry["temp"], entry["hum"],
                             entry["gas"], entry["flame"], entry["status"]])
    except:
        pass


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Backend OK"})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}

    try:
        temp = float(data["temp"])
        hum = float(data["hum"])
        gas = float(data["gas"])
        flame = float(data["flame"])
    except:
        return jsonify({"error": "Invalid input"}), 400

    sample = np.array([[temp, hum, gas, flame]])
    pred = model.predict(sample)[0]
    status = label_encoder.inverse_transform([pred])[0].upper()

    entry = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "temp": temp,
        "hum": hum,
        "gas": gas,
        "flame": flame,
        "status": status
    }

    history.append(entry)
    append_csv(entry)
    if USE_MYSQL:
        insert_row(temp, hum, gas, flame, status)

    return jsonify({"status": status, "entry": entry})


@app.route("/api/save-data", methods=["POST"])
def api_save():
    data = request.form.to_dict() or {}

    try:
        temp = float(data["temperature"])
        hum = float(data["humidity"])
        gas = float(data["gas"])
        flame = float(data["flame"])
        status = data.get("status", "").upper()
    except:
        return jsonify({"error": "Invalid input"}), 400

    entry = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "temp": temp,
        "hum": hum,
        "gas": gas,
        "flame": flame,
        "status": status
    }

    history.append(entry)
    append_csv(entry)
    if USE_MYSQL:
        insert_row(temp, hum, gas, flame, status)

    return jsonify({"saved": True})


@app.route("/latest", methods=["GET"])
def latest():
    return jsonify({
        "last": history[-1] if history else {},
        "history": list(history)
    })


# ========== ONLY BUZZER CONTROL ==========
@app.route("/buzzer/<mode>", methods=["POST"])
def buzzer_set(mode):
    mode = mode.upper()
    if mode not in ("OFF", "WARN", "DANGER"):
        return jsonify({"error": "Invalid mode"}), 400
    buzzer_state["mode"] = mode
    return jsonify({"buzzer": buzzer_state})


@app.route("/device/commands", methods=["GET"])
def get_commands():
    return jsonify({
        "buzzer": buzzer_state["mode"]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
