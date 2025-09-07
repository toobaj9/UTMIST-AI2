#!/usr/bin/env bash
set -e  # exit on any error

# ----------------------------
# Helper function: remove path if exists
# ----------------------------
remove_path() {
    if [ -e "$1" ]; then
        echo "Removing existing $1..."
        rm -rf "$1"
    fi
}

# ----------------------------
# Ensure required tools are installed
# ----------------------------
ensure_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Installing $1..."
        if [ -x "$(command -v apt-get)" ]; then
            apt-get update && apt-get install -y "$2"
        elif [ -x "$(command -v yum)" ]; then
            yum install -y "$2"
        else
            echo "Unsupported package manager. Please install $1 manually."
            exit 1
        fi
    fi
}

ensure_command gdown python3-pip
ensure_command unzip unzip

# ----------------------------
# Download & unzip a Google Drive file
# ----------------------------
download_and_unzip() {
    FILE_ID="$1"
    DEST="$2"
    ZIP_NAME="${DEST}.zip"

    # Remove old files/directories
    remove_path "$DEST"
    remove_path "$ZIP_NAME"

    echo "Downloading $ZIP_NAME..."
    gdown "$FILE_ID" -O "$ZIP_NAME"

    # Check if download is a valid zip
    if unzip -tq "$ZIP_NAME" &> /dev/null; then
        echo "Unzipping $ZIP_NAME..."
        unzip -q "$ZIP_NAME"
        rm "$ZIP_NAME"
    else
        echo "❌ $ZIP_NAME is not a valid zip file!"
        exit 1
    fi
}

# ----------------------------
# Download assets and attacks
# ----------------------------
download_and_unzip 1F2MJQ5enUPVtyi3s410PUuv8LiWr8qCz assets
download_and_unzip 1LAOL8sYCUfsCk3TEA3vvyJCLSl0EdwYB attacks

# ----------------------------
# Install Python dependencies
# ----------------------------
if [ -f requirements.txt ]; then
    echo "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "No requirements.txt found. Skipping Python dependency installation."
fi

echo "✅ Installation complete."
