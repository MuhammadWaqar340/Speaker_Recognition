import shutil
import sys
from flask import Flask, jsonify, request
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
import os
import librosa
import soundfile as sf
from werkzeug.utils import secure_filename
from datetime import datetime
import numpy as np
import noisereduce as nr
from dotenv import load_dotenv
import wave
import json
from vosk import Model, KaldiRecognizer, SpkModel


# Init App
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# MongoDB Configuration
app.config["MONGO_URI"] = os.environ.get("MONGO_URI")
mongo = PyMongo(app)

# Saved File Folder
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER")
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

    request_id = str(ObjectId())
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

    # Insert file metadata into MongoDB
    item = {
        "username": username,
        "folder_id": request_id,
        "files": [
            {"file_path": fp, "filename": fn}
            for fp, fn, ar in zip(filepaths, filenames, analysis_results)
        ],
    }
    result = mongo.db.files.insert_one(item)
    item["_id"] = str(result.inserted_id)

    return jsonify(item), 201


##################################################################################################

# initializing Model
SPK_MODEL_PATH = "./model/vosk-model-spk-0.4"

if not os.path.exists(SPK_MODEL_PATH):
    sys.exit(1)

model = Model(lang="en")
spk_model = SpkModel(SPK_MODEL_PATH)


# Function to calculate cosine distance
def cosine_dist(x, y):
    nx = np.array(x)
    ny = np.array(y)
    return round(1 - np.dot(nx, ny) / np.linalg.norm(nx) / np.linalg.norm(ny), 1)


# Request
@app.route("/api/verify", methods=["POST"])
def verify_speaker():
    if "audio_file" not in request.files:
        return jsonify({"error": "No audio file found"}), 400

    if "folder_id" not in request.form:
        return jsonify({"error": "No folder_id provided"}), 400

    audio_file = request.files["audio_file"]
    folder_id = request.form["folder_id"]

    # Check if the uploaded file has an allowed extension
    if not allowed_file(audio_file.filename):
        return jsonify({"error": "Only audio files are allowed"}), 400

    # Generate a unique filename
    filename = secure_filename(audio_file.filename)
    filename_without_extension, extension = os.path.splitext(filename)
    unique_filename = f"{filename_without_extension}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}{extension}"
    audio_path = os.path.join("/tmp", unique_filename)
    audio_file.save(audio_path)

    # Check if file needs to be converted to WAV
    if not audio_path.lower().endswith(".wav"):
        try:
            audio_data, sr = librosa.load(audio_path, sr=None)
            wav_filename = os.path.splitext(unique_filename)[0] + ".wav"
            wav_filepath = os.path.join("/tmp", wav_filename)
            sf.write(wav_filepath, audio_data, sr)

            # Remove the original non-wav file
            os.remove(audio_path)
            audio_path = wav_filepath
        except Exception as e:
            return jsonify(
                {
                    "error": f"Failed to convert file {audio_file.filename}. Error: {str(e)}"
                }
            ), 500

    # Apply noise reduction
    try:
        noise_reduction(audio_path)
    except Exception as e:
        return jsonify(
            {"error": f"Failed to apply noise reduction. Error: {str(e)}"}
        ), 500

    # Load the audio file
    try:
        wf = wave.open(audio_path, "rb")
    except wave.Error as e:
        return jsonify({"error": f"Failed to load audio file. Error: {str(e)}"}), 400

    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        return jsonify({"error": "Audio file must be WAV format mono PCM"}), 400

    # Initialize recognizer
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetSpkModel(spk_model)

    # Initialize an empty list to store x-vector embeddings for the uploaded file
    x_vectors = []

    # Loop over the audio file and accumulate x-vector embeddings
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            if "spk" in res:
                x_vectors.append(res["spk"])

    # Compute the average x-vector for the uploaded file
    if not x_vectors:
        return jsonify({"error": "No x-vectors found in the audio file"}), 400

    avg_x_vector = np.mean(x_vectors, axis=0)

    # Retrieve file paths from the database
    user_data = mongo.db.files.find_one({"folder_id": folder_id})
    if not user_data:
        return jsonify({"error": "folder_id not found"}), 404

    file_paths = [file["file_path"] for file in user_data["files"]]

    cosine_distances = {}
    valid_distances_count = 0

    # Calculate cosine distances for each file in the folder
    for file_path in file_paths:
        # Load each audio file
        try:
            wf = wave.open(file_path, "rb")
        except wave.Error as e:
            return jsonify(
                {"error": f"Failed to load audio file from folder. Error: {str(e)}"}
            ), 400

        if (
            wf.getnchannels() != 1
            or wf.getsampwidth() != 2
            or wf.getcomptype() != "NONE"
        ):
            return jsonify({"error": "Audio file must be WAV format mono PCM"}), 400

        # Initialize recognizer
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetSpkModel(spk_model)

        # Initialize an empty list to store x-vector embeddings for the file in the folder
        folder_x_vectors = []

        # Loop over the audio file and accumulate x-vector embeddings
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                if "spk" in res:
                    folder_x_vectors.append(res["spk"])

        # Compute the average x-vector for the file in the folder
        if not folder_x_vectors:
            return jsonify(
                {"error": f"No x-vectors found in the audio file: {file_path}"}
            ), 400

        avg_folder_x_vector = np.mean(folder_x_vectors, axis=0)

        # Calculate cosine distance
        speaker_distance = cosine_dist(avg_x_vector, avg_folder_x_vector)
        cosine_distances[file_path] = speaker_distance

        # Count valid distances
        if 0.0 <= speaker_distance <= 0.5:
            valid_distances_count += 1

    verification_result = (
        "Successfully Verified!"
        if valid_distances_count >= 2
        else "Speaker Not Verified"
    )

    response = {
        "verification_result": verification_result,
        "cosine_distances": cosine_distances,
    }

    return jsonify(response), 200


#####################################################################################################

# Run Server
if __name__ == "__main__":
    app.run(debug=True)
#####################################################################################################

# import os
# import shutil
# import wave
# import json
# import numpy as np
# from flask import Flask, jsonify, request
# from flask_pymongo import PyMongo
# from bson.objectid import ObjectId
# import librosa
# from werkzeug.utils import secure_filename
# from datetime import datetime
# from dotenv import load_dotenv
# from vosk import Model, KaldiRecognizer, SpkModel
# import soundfile as sf

# # Init App
# app = Flask(__name__)

# # Load environment variables from .env file
# load_dotenv()

# # MongoDB Configuration
# app.config["MONGO_URI"] = os.environ.get("MONGO_URI")
# mongo = PyMongo(app)

# # Saved File Folder
# UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER")
# if UPLOAD_FOLDER and not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# ALLOWED_EXTENSIONS = {
#     "wav",
#     "mp3",
#     "ogg",
#     "flac",
#     "aac",
#     "m4a",
#     "wma",
#     "aiff",
#     "au",
#     "amr",
#     "ape",
#     "dss",
#     "dsf",
#     "dvf",
#     "gsm",
#     "iklax",
#     "ivs",
#     "mka",
#     "mmf",
#     "mpc",
#     "msv",
#     "opus",
#     "ra",
#     "rm",
#     "sln",
#     "tta",
#     "vox",
#     "wv",
# }

# # Path to Vosk model and speaker model

# SPK_MODEL_PATH = "./model/vosk-model-spk-0.4"

# if not os.path.exists(SPK_MODEL_PATH) or not os.path.exists(SPK_MODEL_PATH):
#     raise FileNotFoundError(
#         "Please download the Vosk model and speaker model from https://alphacephei.com/vosk/models and unpack them into 'model' and 'model-spk' directories respectively."
#     )

# model = Model(lang="en")
# spk_model = SpkModel(SPK_MODEL_PATH)


# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# def cosine_dist(x, y):
#     nx = np.array(x)
#     ny = np.array(y)
#     return round(1 - np.dot(nx, ny) / np.linalg.norm(nx) / np.linalg.norm(ny), 1)


# # POST Api for File Upload
# @app.route("/api/upload", methods=["POST"])
# def upload_file():
#     if "username" not in request.form:
#         return jsonify({"error": "Username is required"}), 400

#     username = request.form["username"]
#     files = request.files.getlist("files")

#     for file in files:
#         if not allowed_file(file.filename):
#             return jsonify({"error": "Only audio files are allowed"}), 400

#     if len(files) != 3:
#         return jsonify({"error": "Exactly 3 audio files are required"}), 400

#     request_id = str(ObjectId())
#     user_upload_folder = os.path.join(UPLOAD_FOLDER, request_id)
#     os.makedirs(user_upload_folder)

#     filepaths = []
#     filenames = []
#     long_duration_files = []

#     for file in files:
#         if file.filename == "":
#             return jsonify({"error": "One or more files have no selected file"}), 400
#         if file:
#             try:
#                 # Generate a unique filename
#                 filename = secure_filename(file.filename)
#                 filename_without_extension, extension = os.path.splitext(filename)
#                 unique_filename = f"{filename_without_extension}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}{extension}"
#                 filepath = os.path.join(user_upload_folder, unique_filename)
#                 file.save(filepath)

#                 # Check file format and convert if necessary
#                 if not filename.lower().endswith(".wav"):
#                     try:
#                         audio_data, sr = librosa.load(filepath, sr=None)
#                         wav_filename = os.path.splitext(unique_filename)[0] + ".wav"
#                         wav_filepath = os.path.join(user_upload_folder, wav_filename)
#                         sf.write(wav_filepath, audio_data, sr)

#                         # Remove the original non-wav file
#                         os.remove(filepath)
#                         filepath = wav_filepath
#                         unique_filename = wav_filename
#                     except Exception as e:
#                         return jsonify(
#                             {
#                                 "error": f"Failed to convert file {file.filename}. Error: {str(e)}"
#                             }
#                         ), 500

#                 # Load the audio file to check duration
#                 y, sr = librosa.load(filepath, sr=None)
#                 duration = librosa.get_duration(y=y, sr=sr)

#                 # Check if the file exceeds 30 seconds
#                 if duration > 30:
#                     long_duration_files.append(file.filename)

#                 filepaths.append(filepath)
#                 filenames.append(unique_filename)
#             except Exception as e:
#                 return jsonify(
#                     {"error": f"Failed to save file {file.filename}. Error: {str(e)}"}
#                 ), 500

#     if long_duration_files:
#         shutil.rmtree(user_upload_folder)
#         return jsonify(
#             {
#                 "error": f"The following files exceed 30 seconds: [{', '.join(long_duration_files)}]. Upload failed."
#             }
#         ), 400

#     # Insert file metadata into MongoDB
#     item = {
#         "username": username,
#         "folder_id": request_id,
#         "files": [
#             {"file_path": fp, "filename": fn} for fp, fn in zip(filepaths, filenames)
#         ],
#     }
#     result = mongo.db.files.insert_one(item)
#     item["_id"] = str(result.inserted_id)

#     return jsonify(item), 201


# # POST Api for Speaker Verification
# @app.route("/api/verify_speaker", methods=["POST"])
# def verify_speaker():
#     if "file" not in request.files:
#         return jsonify({"error": "No file provided"}), 400

#     file = request.files["file"]
#     if not allowed_file(file.filename):
#         return jsonify({"error": "Only audio files are allowed"}), 400

#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400

#     user_upload_folder = os.path.join(UPLOAD_FOLDER, "verification")
#     if not os.path.exists(user_upload_folder):
#         os.makedirs(user_upload_folder)

#     filename = secure_filename(file.filename)
#     filepath = os.path.join(user_upload_folder, filename)
#     file.save(filepath)

#     try:
#         wf = wave.open(filepath, "rb")
#         if (
#             wf.getnchannels() != 1
#             or wf.getsampwidth() != 2
#             or wf.getcomptype() != "NONE"
#         ):
#             return jsonify({"error": "Audio file must be WAV format mono PCM."}), 400

#         rec = KaldiRecognizer(model, wf.getframerate())
#         rec.SetSpkModel(spk_model)

#         x_vectors = []

#         while True:
#             data = wf.readframes(4000)
#             if len(data) == 0:
#                 break
#             if rec.AcceptWaveform(data):
#                 res = json.loads(rec.Result())
#                 if "spk" in res:
#                     x_vectors.append(res["spk"])

#         if not x_vectors:
#             return jsonify({"error": "No speaker data found in the audio file."}), 400

#         avg_x_vector = np.mean(x_vectors, axis=0)

#         # Retrieve folder ID from the request body
#         folder_id = request.form.get("folder_id")
#         if not folder_id:
#             return jsonify({"error": "Folder ID is required"}), 400

#         # Fetch the reference speaker signature based on the folder ID from the database
#         folder_data = mongo.db.files.find_one({"folder_id": folder_id})
#         if not folder_data:
#             return jsonify({"error": "Folder not found"}), 404

#         # Extract x-vectors of audio files within the folder
#         folder_x_vectors = []
#         for audio_file_data in folder_data["files"]:
#             audio_filepath = audio_file_data["file_path"]
#             audio_wf = wave.open(audio_filepath, "rb")
#             audio_rec = KaldiRecognizer(model, audio_wf.getframerate())
#             audio_rec.SetSpkModel(spk_model)

#             while True:
#                 audio_data = audio_wf.readframes(4000)
#                 if len(audio_data) == 0:
#                     break
#                 if audio_rec.AcceptWaveform(audio_data):
#                     audio_res = json.loads(audio_rec.Result())
#                     if "spk" in audio_res:
#                         folder_x_vectors.append(audio_res["spk"])

#         if not folder_x_vectors:
#             return jsonify(
#                 {"error": "No speaker data found in the reference audio files."}
#             ), 400

#         avg_folder_x_vector = np.mean(folder_x_vectors, axis=0)

#         # Calculate cosine distance between the average x-vectors
#         speaker_distance = cosine_dist(avg_folder_x_vector, avg_x_vector)
#         verification_result = (
#             "Successfully Verified!"
#             if 0.0 <= speaker_distance <= 0.5
#             else "You are not Verified!"
#         )

#         return jsonify(
#             {
#                 "speaker_distance": speaker_distance,
#                 "verification_result": verification_result,
#             }
#         )
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     app.run(debug=True)
