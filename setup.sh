#!/usr/bin/env bash

echo "HomeValue-Analytics Installer"
echo "==========================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python is not installed! Please install Python 3.12 or newer."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Build executable
echo "Building executable..."
pyinstaller --onefile --noconsole --clean --add-data "src:src" src/app.py

# Ask about shortcut creation
read -p "Do you want to create a shortcut? (y/N): " create_shortcut
if [[ $create_shortcut =~ ^[Yy]$ ]]; then
    read -p "Enter shortcut location (e.g. ~/Desktop): " shortcut_path
    shortcut_path="${shortcut_path/#\~/$HOME}"
    
    # Create .desktop file
    cat > "$shortcut_path/HomeValue-Analytics.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=HomeValue-Analytics
Comment=Analytics tool for housing prices
Exec=$(pwd)/dist/HomeValue-Analytics
Icon=chart-line
Terminal=false
Categories=Office;
EOF
    
    # Make executable
    chmod +x "$shortcut_path/HomeValue-Analytics.desktop"
    echo "Shortcut created at $shortcut_path"
fi

echo "Installation completed!"
echo "Run the application from dist/HomeValue-Analytics"
read -p "Press Enter to exit..."