#!/usr/bin/env bash
set -e  # exit if any command fails

# Function to remove file or directory if it exists
remove_path() {
    if [ -e "$1" ]; then
        rm -rf "$1"
    fi
}

# Make sure gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

# 1. Remove old assets
remove_path assets
remove_path assets.zip

# 2. Download & unzip assets
echo "Downloading assets.zip..."
gdown "https://drive.google.com/file/d/1F2MJQ5enUPVtyi3s410PUuv8LiWr8qCz/view?usp=sharing" -O assets.zip
echo "Unzipping assets.zip..."
unzip -q assets.zip
rm assets.zip

# 3. Remove old attacks
remove_path attacks
remove_path attacks.zip

# 4. Download & unzip attacks
echo "Downloading attacks.zip..."
gdown "https://drive.google.com/file/d/1LAOL8sYCUfsCk3TEA3vvyJCLSl0EdwYB/view?usp=sharing" -O attacks.zip
echo "Unzipping attacks.zip..."
unzip -q attacks.zip
rm attacks.zip

# 5. Install dependencies
if [ -f requirements.txt ]; then
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "No requirements.txt found. Skipping dependency installation."
fi

echo "✅ Installation complete."
