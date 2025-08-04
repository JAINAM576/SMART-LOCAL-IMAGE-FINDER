import gzip
import shutil
import os
import requests
import zipfile
import sys
import time
import json

# Log file path
LOG_FILE_PATH = '../assests/log.json'

# GitHub uploaded files
urls = [
    "https://github.com/JAINAM576/MODELS_FORLOCAL/releases/download/BLIP/blip_decoder_opt.onnx.gz",
    "https://github.com/JAINAM576/MODELS_FORLOCAL/releases/download/BLIP/blip_vision_opt.onnx.gz",
    "https://github.com/JAINAM576/MODELS_FORLOCAL/releases/download/BLIP/tokenizer_config.json.gz",
    "https://github.com/JAINAM576/MODELS_FORLOCAL/releases/download/v1.0/embedd_model_MiniLm.onnx.gz",
    "https://github.com/JAINAM576/MODELS_FORLOCAL/releases/download/v1.0e/all-MiniLM-L6-v2.zip"
]

# ZIP MODEL PATHS
blip_text_decoder_zip = "../models/blip-image-captioning/blip_decoder_opt.onnx.gz"
blip_vision_zip = "../models/blip-image-captioning/blip_vision_opt.onnx.gz"
tokenizer_config_zip = "../models/blip-image-captioning/tokenizer_config.json.gz"
embed_model_zip = "../models/embedd_model/embedd_model_MiniLm.onnx.gz"
sentance_model_zip = "../models/all-MiniLM-L6-v2.zip"
sentance_model = "../models/all-MiniLM-L6-v2"

# Model Path Map
modelPathMap = {
    blip_text_decoder_zip: 0,
    blip_vision_zip: 1,
    tokenizer_config_zip: 2,
    embed_model_zip: 3,
    sentance_model_zip: 4
}

modelPathMapName = {
    blip_text_decoder_zip: "Text Encoder",
    blip_vision_zip: "Vision Model",
    tokenizer_config_zip: "Tokenizer",
    embed_model_zip: "Embedding Model",
    sentance_model_zip: "Sentence Model"
}

# UNZIP MODEL PATHS
blip_text_decoder = "../models/blip-image-captioning/blip_decoder_opt.onnx"
blip_vision = "../models/blip-image-captioning/blip_vision_opt.onnx"
tokenizer_config = "../models/blip-image-captioning/tokenizer_config.json"
embed_onnx_model = "../models/embedd_model/embedd_model_MiniLm.onnx"

# Utility to load log.json
def load_log():
    if not os.path.exists(LOG_FILE_PATH):
        return {}
    with open(LOG_FILE_PATH, 'r') as f:
        return json.load(f)

def update_log(model_zip):
    log = load_log()
    log[model_zip] = True
    with open(LOG_FILE_PATH, 'w') as f:
        json.dump(log, f)

def is_model_fully_downloaded(model_path, model_zip):
    log = load_log()
    if os.path.exists(model_path):
        return True
    elif os.path.exists(model_zip) and not log.get(model_zip):
        print(f"Incomplete file found for {modelPathMapName[model_zip]}, deleting...")
        os.remove(model_zip)
        return False
    else:
        return False

# Download function
def downloadModel(dirpath, index):
    response = requests.get(urls[index], stream=True)
    total_size = int(response.headers.get('content-length', 0))
    bytes_downloaded = 0
    chunk_size = 8192
    last_update_time = time.time()

    with open(f"{dirpath}", 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bytes_downloaded += len(chunk)

                percent = (bytes_downloaded / total_size) * 100

                if time.time() - last_update_time > 0.2:
                    print(f"PROGRESS:{percent:.2f}|NAME:{modelPathMapName[dirpath]}")
                    sys.stdout.flush()
                    last_update_time = time.time()

                print("RUNNING")
                sys.stdout.flush()

    print("Download Completed")
    update_log(dirpath)

# Unzip functions
def needToUnzip(model_path):
    output_path = model_path[:-3]
    with gzip.open(model_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def unzip_folder(zip_path, extract_to_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_folder)

# Main model check
def checkModelExist(model_path, model_zip):
    if is_model_fully_downloaded(model_path, model_zip):
        return
    else:
        if os.path.exists(model_zip):
            try:
                if model_zip == sentance_model_zip:
                    unzip_folder(model_zip, model_path)
                else:
                    needToUnzip(model_zip)
                update_log(model_zip)
            except Exception as e:
                print("ERROR DURING UNZIP", e)
        else:
            downloadModel(model_zip, modelPathMap[model_zip])
            checkModelExist(model_path, model_zip)

def setup_all_models():
    checkModelExist(blip_vision, blip_vision_zip)
    checkModelExist(blip_text_decoder, blip_text_decoder_zip)
    checkModelExist(tokenizer_config, tokenizer_config_zip)
    checkModelExist(embed_onnx_model, embed_model_zip)
    checkModelExist(sentance_model, sentance_model_zip)
    print("ALL_DOWNLOADS_COMPLETED")
    sys.stdout.flush()

if __name__ == "__main__":
    setup_all_models()
