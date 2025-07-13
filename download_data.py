import os
import requests
from pathlib import Path

def download_from_google_drive(file_id, destination):
    print("⏬ Downloading studentVle.csv...")
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(destination, "wb") as f:
            for chunk in response.iter_content(32768):
                f.write(chunk)
        print("✅ Download complete:", destination)
    else:
        print("❌ Failed to download. Status code:", response.status_code)

def main():
    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)
    file_path = "data/studentVle.csv"

    if not os.path.exists(file_path):
        file_id = "13U_1xVL2C1ucei9Q8C1TSqWwiePZ4qRc"  # Your file ID
        download_from_google_drive(file_id, file_path)
    else:
        print("✅ studentVle.csv already exists.")

if __name__ == "__main__":
    main()
