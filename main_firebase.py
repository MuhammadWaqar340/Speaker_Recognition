import shutil
from flask import Flask, jsonify, request
import os
import librosa
import soundfile as sf
from werkzeug.utils import secure_filename
from datetime import datetime
import numpy as np
import noisereduce as nr
import firebase_admin
from firebase_admin import credentials, db
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Init App
app = Flask(__name__)

# Firebase Configuration
cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
database_url = os.getenv("FIREBASE_DATABASE_URL")

cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(
    cred,
    {"databaseURL": database_url},
)

# Saved File Folder
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER")
if UPLOAD_FOLDER and not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {
    "wav",
    "mp3",
    "ogg",
    "flac",
    "aac",
    "m4a",
    "wma",
    "aiff",
    "au",
    "amr",
    "ape",
    "dss",
    "dsf",
    "dvf",
    "gsm",
    "iklax",
    "ivs",
    "mka",
    "mmf",
    "mpc",
    "msv",
    "opus",
    "ra",
    "rm",
    "sln",
    "tta",
    "vox",
    "wv",
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_noise_or_silence(filepath):
    y, sr = librosa.load(filepath, sr=None)
    rms = librosa.feature.rms(y=y)[0]
    silence_threshold = 0.01
    noise_threshold = 0.1

    silence_percentage = np.sum(rms < silence_threshold) / len(rms)
    noise_percentage = np.sum(rms > noise_threshold) / len(rms)

    silence_percentage = round(silence_percentage, 1)
    noise_percentage = round(noise_percentage, 1)

    if silence_percentage > 0.7:
        return "silent"
    elif noise_percentage > 0.5:
        return "noisy"
    else:
        return "normal"


def noise_reduction(filepath):
    y, sr = librosa.load(filepath, sr=None)
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    sf.write(filepath, reduced_noise, sr)


# POST Api
@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "username" not in request.form:
        return jsonify({"error": "Username is required"}), 400

    username = request.form["username"]
    files = request.files.getlist("files")

    for file in files:
        if not allowed_file(file.filename):
            return jsonify({"error": "Only audio files are allowed"}), 400

    if len(files) != 3:
        return jsonify({"error": "Exactly 3 audio files are required"}), 400

    request_id = str(uuid.uuid4())
    user_upload_folder = os.path.join("uploads", request_id)
    os.makedirs(user_upload_folder)

    filepaths = []
    filenames = []
    analysis_results = []
    long_duration_files = []
    silent_files = []

    for file in files:
        if file.filename == "":
            return jsonify({"error": "One or more files have no selected file"}), 400
        if file:
            try:
                # Generate a unique filename
                filename = secure_filename(file.filename)
                filename_without_extension, extension = os.path.splitext(filename)
                unique_filename = f"{filename_without_extension}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}{extension}"
                filepath = os.path.join(user_upload_folder, unique_filename)
                file.save(filepath)

                # Check file format and convert if necessary
                if not filename.lower().endswith(".wav"):
                    try:
                        audio_data, sr = librosa.load(filepath, sr=None)
                        wav_filename = os.path.splitext(unique_filename)[0] + ".wav"
                        wav_filepath = os.path.join(user_upload_folder, wav_filename)
                        sf.write(wav_filepath, audio_data, sr)

                        # Remove the original non-wav file
                        os.remove(filepath)
                        filepath = wav_filepath
                        unique_filename = wav_filename
                    except Exception as e:
                        return jsonify(
                            {
                                "error": f"Failed to convert file {file.filename}. Error: {str(e)}"
                            }
                        ), 500

                # Load the audio file to check duration
                y, sr = librosa.load(filepath, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)

                # Check if the file exceeds 30 seconds
                if duration > 30:
                    long_duration_files.append(file.filename)

                # Apply noise reduction to the file
                noise_reduction(filepath)

                # Analyze the file for silence
                analysis_result = detect_noise_or_silence(filepath)

                # Check if the file is silent
                if analysis_result == "silent":
                    silent_files.append(file.filename)

                analysis_results.append(analysis_result)

                filepaths.append(filepath)
                filenames.append(unique_filename)
            except Exception as e:
                return jsonify(
                    {"error": f"Failed to save file {file.filename}. Error: {str(e)}"}
                ), 500

    if long_duration_files:
        shutil.rmtree(user_upload_folder)
        return jsonify(
            {
                "error": f"The following files exceed 30 seconds: [{', '.join(long_duration_files)}]. Upload failed."
            }
        ), 400

    if silent_files:
        shutil.rmtree(user_upload_folder)
        return jsonify(
            {
                "error": f"The following files are silent or have no clear speaker voice: [{', '.join(silent_files)}]. Upload failed."
            }
        ), 400

    # Insert file metadata into Firebase Realtime Database
    item = {
        "username": username,
        "folder_id": request_id,
        "files": [
            {"file_path": fp, "filename": fn}
            for fp, fn, ar in zip(filepaths, filenames, analysis_results)
        ],
    }
    ref = db.reference("files")
    new_ref = ref.push(item)
    item["id"] = new_ref.key

    return jsonify(item), 201


# Run Server
if __name__ == "__main__":
    app.run(debug=True)
