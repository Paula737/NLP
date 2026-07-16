import csv
import os

DATABASE_FILE = "database.csv"


def initialize_database():
    if not os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Input", "Prediction"])


def save_record(user_input, prediction):
    with open(DATABASE_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([user_input, prediction])


def read_database():
    initialize_database()

    with open(DATABASE_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)