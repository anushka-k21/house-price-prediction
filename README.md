# Multimodal Price Prediction Project

This repository contains the codebase for a **regression-based price prediction system** using:
- **Tabular data**
- **Image data**
- **Multimodal learning (Tabular + Image)**

The project explores and compares different modeling approaches, including classical ML models, CNN-based regression, and fine-tuned multimodal models.

---

## Project Structure

```text

├── data_fetcher.py         # Script to fetch / load data
├── data_fetcher_test.py    # Tests for data fetching logic
│
├── preprocessing.ipynb     # Data cleaning & preprocessing notebook
│
├── model_training-1.ipynb  # Tabular regression models
├── model_training-2.ipynb  # Image-only CNN regression model
├── model_training-3.ipynb  # Multimodal regression & fine-tuned model
│
├── 24116045_final.csv     # Final results / predictions
├── 24116045_report.pdf     # Project report
│
├── .gitignore              # Files excluded from version control
└── README.md               # Project documentation
```

## Prerequisites

Make sure the following are installed on your system:

- Python **3.9 or higher**
- pip
- Git
- Jupyter Notebook

### Required Libraries
- numpy  
- pandas  
- scikit-learn  
- xgboost  
- torch  
- torchvision  
- opencv-python  
- matplotlib  
- seaborn  
- tqdm  
- sentinelhub  


##  Setup Instructions

### 1. Clone Repository
```bash
git clone <repository-url>
cd <project-folder>
```
2. Create Virtual Environment
``` bash
python -m venv venv
```
3. Install Dependencies
``` bash
pip install numpy pandas scikit-learn xgboost torch torchvision \
            matplotlib seaborn opencv-python tqdm sentinelhub
```
4- Sentinel Hub API Setup (Satellite Imagery)
Step 1: Create Account
Sign up at: https://www.sentinel-hub.com

Step 2: Create OAuth Client
Go to Dashboard → User Settings → OAuth Clients
Create a new client
Note the following:
CLIENT_ID
CLIENT_SECRET

Step 3: Create Configuration File
Create the file:
config/sentinel_config.py
```python
from sentinelhub import SHConfig
config = SHConfig()
config.sh_client_id = "YOUR_CLIENT_ID"
config.sh_client_secret = "YOUR_CLIENT_SECRET"
config.instance_id = "YOUR_INSTANCE_ID"  # if applicable
```
## Data Preparation
Tabular Data
Place processed CSV files here:
```
data/processed/
    ├── train_clean.csv
    └── test_clean.csv
```
Satellite Images
Downloaded images are saved to:
```
data/images/sentinel/
```
Each image corresponds to a dataset row and is named by index:
```
0.png, 1.png, 2.png, ...
```
 ### Download Satellite Images
Run:
```
python data_fetcher.py
```
