# Key parameters:
# Resolution: 10m
# Size: 256x256
# Time range: last 2–3 years
# Cloud coverage: <20%

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentinelhub import (
    SentinelHubRequest,
    DataCollection,
    MimeType,
    CRS,
    BBox,
    bbox_to_dimensions
)
from config.sentinel_config import config


#CONFIG
DATA_CSV = "data/processed/train_clean.csv"
IMAGE_DIR = "data/images/sentinel"
RESOLUTION = 10        # meters per pixel (Sentinel-2 RGB)
IMG_SIZE = 256         # final image size (256x256)
TIME_RANGE = ("2022-01-01", "2023-12-31")
MAX_CLOUD = 20         # %

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

    # Define bounding box
    bbox = BBox(
        bbox=[
            lon - 0.01,
            lat - 0.01,
            lon + 0.01,
            lat + 0.01
        ],
        crs=CRS.WGS84
    )

    size = bbox_to_dimensions(bbox, resolution=RESOLUTION)

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
        size=size,
        config=config
    )

    image = request.get_data(save_data=False)[0]

    if image is not None:
        from PIL import Image
        img = Image.fromarray(image)
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img.save(save_path)


# DOWNLOAD LOOP
MAX_IMAGES = 500   

for idx, row in tqdm(df.iterrows(), total=len(df)):

    if idx >= MAX_IMAGES:
        break

    img_path = os.path.join(IMAGE_DIR, f"{idx}.png")
    if os.path.exists(img_path):
        continue  #Shifts already downloaded image
    try:
        fetch_sentinel_image(row["lat"], row["long"], img_path)
    except Exception as e:
        print(f"Error at index {idx}: {e}")






































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
