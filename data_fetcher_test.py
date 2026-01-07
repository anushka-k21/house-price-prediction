# Key parameters:
# Resolution: 10m
# Size: 256x256
# Time range: last 2–3 years
# Cloud coverage: <20%
##
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from sentinelhub import (
    SentinelHubRequest,
    DataCollection,
    MimeType,
    CRS,
    BBox,
    bbox_to_dimensions
)
from config.sentinel_config import config
from PIL import Image

from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_THREADS = 6  

#CONFIG
DATA_CSV = "data/processed/test_clean.csv"
SPLIT = "test"  
IMAGE_DIR = f"data/images/sentinel_{SPLIT}"
os.makedirs(IMAGE_DIR, exist_ok=True)
RESOLUTION = 10        # meters per pixel (Sentinel-2 RGB)
IMG_SIZE = 256         # final image size (256x256)
TIME_RANGE = ("2022-01-01", "2023-12-31")
MAX_CLOUD = 20         # %
###
MAX_THREADS = 8        # Number of parallel downloads
RETRIES = 3            # Retry failed downloads
DELAY = 5              # Seconds between retries

os.makedirs(IMAGE_DIR, exist_ok=True)

# LOAD DATA
df = pd.read_csv(DATA_CSV)


# EVALSCRIPT (RGB only)
evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04"], //Blue, Green, Red
    output: { bands: 3 }
  };
}

function evaluatePixel(sample) {
  return [sample.B04, sample.B03, sample.B02]; // RGB
}
"""
def fetch_sentinel_image(lat, lon, save_path):
    """
    Fetch Sentinel-2 RGB image around (lat, lon)
    """

    # Half-size in degrees for 256x256 @ 10m
    half_size = (IMG_SIZE * RESOLUTION) / 2 / 111320

    bbox = BBox(
        bbox=[
            lon - half_size,
            lat - half_size,
            lon + half_size,
            lat + half_size
        ],
        crs=CRS.WGS84
    )

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=TIME_RANGE,
                mosaicking_order="leastCC",
                maxcc=MAX_CLOUD / 100
            )
        ],
        responses=[
            SentinelHubRequest.output_response("default", MimeType.PNG)
        ],
        bbox=bbox,
        size=(IMG_SIZE, IMG_SIZE),  # DIRECT 256x256
        config=config
    )

    image = request.get_data(save_data=False)[0]

    if image is None:
        raise ValueError("No image returned")

    Image.fromarray(image).save(save_path)

def process_row(row):
    img_id = row.name  # index
    img_path = os.path.join(IMAGE_DIR, f"{img_id}.png")

    if os.path.exists(img_path):
        return f"{img_id} SKIPPED"

    success = fetch_with_retry(
        row["lat"],
        row["long"],
        img_path,
        retries=RETRIES,
        delay=DELAY
    )

    return f"{img_id} {'OK' if success else 'FAIL'}"

# def fetch_sentinel_image(lat, lon, save_path):
#     """
#     Fetch Sentinel-2 RGB image around (lat, lon)
#     """

#     # Define bounding box
#     bbox = BBox(
#         bbox=[
#             lon - 0.01,
#             lat - 0.01,
#             lon + 0.01,
#             lat + 0.01
#         ],
#         crs=CRS.WGS84
#     )

#     size = bbox_to_dimensions(bbox, resolution=RESOLUTION)

#     request = SentinelHubRequest(
#         evalscript=evalscript,
#         input_data=[
#             SentinelHubRequest.input_data(
#                 data_collection=DataCollection.SENTINEL2_L2A,
#                 time_interval=TIME_RANGE,
#                 mosaicking_order="leastCC",
#                 maxcc=MAX_CLOUD / 100
#             )
#         ],
#         responses=[
#             SentinelHubRequest.output_response("default", MimeType.PNG)
#         ],
#         bbox=bbox,
#         size=size,
#         config=config
#     )

#     image = request.get_data(save_data=False)[0]

#     if image is not None:
#         from PIL import Image
#         img = Image.fromarray(image)
#         img = img.resize((IMG_SIZE, IMG_SIZE))
#         img.save(save_path)
#     else:
#         raise ValueError("No image returned by SentinelHub")

def fetch_with_retry(lat, lon, save_path, retries=3, delay=5):
    """Retry logic with delay"""
    for attempt in range(retries):
        try:
            fetch_sentinel_image(lat, lon, save_path)
            return True
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {save_path}: {e}")
            time.sleep(delay)
    print(f"Failed to download {save_path} after {retries} attempts.")
    return False

# LOG_FILE = "download_log_test.txt"

# with open(LOG_FILE, "a") as log_file:
#     for idx, row in tqdm(df.iterrows(), total=len(df)):

#         img_path = os.path.join(IMAGE_DIR, f"{idx}.png")

#         # Skip if already exists (partial download check)
#         if os.path.exists(img_path):
#             log_file.write(f"{idx} SKIPPED (exists)\n")
#             continue

#         success = fetch_with_retry(row["lat"], row["long"], img_path)
#         if success:
#             log_file.write(f"{idx} OK\n")
#         else:
#             log_file.write(f"{idx} FAIL\n")
        
LOG_FILE = "download_log_test.txt"

tasks = []

with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    for _, row in df.iterrows():
        tasks.append(executor.submit(process_row, row))

    with open(LOG_FILE, "a") as log_file:
        for future in tqdm(as_completed(tasks), total=len(tasks)):
            log_file.write(future.result() + "\n")


# def fetch_with_retry(lat, lon, save_path, retries=RETRIES, delay=DELAY):
#     """Retry logic for a single image"""
#     for attempt in range(retries):
#         try:
#             fetch_sentinel_image(lat, lon, save_path)
#             return True
#         except Exception as e:
#             print(f"Attempt {attempt+1} failed for {save_path}: {e}")
#             time.sleep(delay)
#     return False

# def download_image(idx_row):
#     """Worker function for multithreading"""
#     idx, row = idx_row
#     img_path = os.path.join(IMAGE_DIR, f"{idx}.png")

#     # Skip if already exists
#     if os.path.exists(img_path):
#         return f"{idx} SKIPPED"

#     success = fetch_with_retry(row["lat"], row["long"], img_path)
#     return f"{idx} {'OK' if success else 'FAIL'}"

# # ---------------- MULTITHREADED DOWNLOAD ----------------
# if __name__ == "__main__":
#     with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
#         futures = [executor.submit(download_image, (idx, row)) for idx, row in df.iterrows()]

#         # Process as they complete
#         for future in tqdm(as_completed(futures), total=len(df)):
#             result = future.result()
#             with open(LOG_FILE, "a") as log_file:
#                 log_file.write(result + "\n")



































# import os
# import time
# import requests
# import pandas as pd
# from tqdm import tqdm

# # CONFIG
# API_KEY = "GNCkSda6NpyRmMfdoDPq"
# # "eFhYtEYgxYlqNpXq7CkC"

# STYLE = "satellite-v2"
# ZOOM = 18
# IMG_SIZE = "256x256"
# FORMAT = "png"

# DATA_CSV = "data/processed/train_clean.csv"
# IMAGE_DIR = "data/images"

# SLEEP_TIME = 0.1  # to avoid rate limits


# # CREATE IMAGE DIRECTORY
# os.makedirs(IMAGE_DIR, exist_ok=True)
# # LOAD DATA
# df = pd.read_csv(DATA_CSV)


# # MAPTILER URL BUILDER
# def build_maptiler_url(lat, lon):
#     return (
#         f"https://api.maptiler.com/maps/{STYLE}/static/"
#         f"{lon},{lat},{ZOOM}/"
#         f"{IMG_SIZE}.{FORMAT}"
#         f"?key={API_KEY}"
#     )

# print(build_maptiler_url(df.iloc[0]["lat"], df.iloc[0]["long"]))

# # MAX_IMAGES = 500
# # # DOWNLOAD LOOP
# # for idx, row in tqdm(df.iterrows(), total=len(df)):
# #     img_path = os.path.join(IMAGE_DIR, f"{idx}.png")
# #     # Skip if already downloaded
# #     if idx >= MAX_IMAGES:
# #         break
# #     if os.path.exists(img_path):
# #         continue
# #     url = build_maptiler_url(row["lat"], row["long"])
# #     try:
# #         response = requests.get(url, timeout=10)
# #         if response.status_code == 200:
# #             with open(img_path, "wb") as f:
# #                 f.write(response.content)
# #         else:
# #             print(f"Failed for index {idx}: {response.status_code}")
# #         time.sleep(SLEEP_TIME)
# #     except Exception as e:
# #         print(f"Error at index {idx}: {e}")
