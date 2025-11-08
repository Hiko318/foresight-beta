# Argus

A multi-window phone mirroring and SAR (Search and Rescue) detection application built with Electron. This beta version separates functionality into multiple windows that group together for better modularity.

## Features

- **Main Control Panel**: Central hub for managing all operations
- **Phone Mirroring**: Uses scrcpy for real-time phone screen mirroring
- **SAR Mode**: YOLO-based object detection for search and rescue operations
- **Window Grouping**: All windows minimize/restore together with the main app
- **Real-time Console**: Live logging and status updates

## Prerequisites

### For Phone Mirroring (scrcpy)
1. Install scrcpy: https://github.com/Genymobile/scrcpy
2. Enable USB Debugging on your Android device
3. Connect your phone via USB

### For YOLO Detection (Optional)
1. Install Python 3.7+
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. YOLO model (yolov8n.pt) is already included for object detection

## Installation

1. Clone or download this repository
2. Install Node.js dependencies:
   ```bash
   npm install
   ```
3. Install Python dependencies (for SAR mode):
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 🚀 Setup Options

#### Option 1: Setup Wizard (Recommended) 🎯
For the best experience, use our professional setup wizard with a graphical interface:

```bash
# Double-click this file:
Setup.bat
```

The setup wizard provides:
- **Professional GUI** with Next/Back navigation
- **Automatic dependency detection** (Node.js, Python, scrcpy)
- **Progress tracking** with visual feedback
- **Step-by-step installation** process
- **Launch option** when setup completes

#### Option 2: Command-Line Installers
For advanced users who prefer command-line installation:

```bash
# Batch installer:
Argus Installer.bat

# PowerShell installer:
Argus Installer.ps1
```

### ⚡ Quick Start (After Setup)
Once dependencies are installed, use these for quick launching:
- **`start-argus.bat`** - Simple batch file launcher
- **`start-argus.ps1`** - PowerShell launcher (alternative)

### 🛠️ Manual Commands
```bash
# Manual start
npm start

# Development mode
npm run dev

# Install dependencies manually
npm install
pip install -r requirements.txt
```

### Production Mode
```bash
npm start
```

### Building the Application
```bash
npm run build
```

## Training a Custom Human Detector (COCO + VisDrone + KAIST)

Foresight uses Ultralytics YOLOv8 for detection. You can train a person-only model by merging COCO2017, VisDrone2019-DET, and KAIST Multispectral Pedestrian datasets.

### Setup
- Python 3.9+ with CUDA-capable GPU recommended
- Install dependencies:
  ```bash
  pip install ultralytics pycocotools opencv-python tqdm
  ```

### Prepare datasets
1. COCO2017: download `train2017`, `val2017`, and `annotations`.
2. VisDrone2019-DET: ensure `images/train`, `images/val`, `annotations/train`, `annotations/val`.
3. KAIST: arrange as
   - `images/train`, `images/val`
   - `annotations/train`, `annotations/val` containing txt files with lines: `filename x y w h` (person boxes).

Run the converter:
```bash
py -3 foresight-beta/scripts/prepare_human_datasets.py ^
  --coco-root C:\data\coco2017 ^
  --visdrone-root C:\data\VisDrone2019-DET ^
  --kaist-root C:\data\KAIST ^
  --out-root C:\data
```

This builds `C:\data\datasets\foresight-human` in YOLO format.

### Train YOLOv8
```bash
python -m ultralytics yolo detect/train ^
  data=foresight-beta/datasets/foresight-human.yaml ^
  model=yolov8n.yaml imgsz=640 epochs=50 batch=16 device=0
```

Tips:
- KAIST thermal/grayscale images are expanded to 3-channel automatically.
- VisDrone categories used: pedestrian(1), person(4), people(5).
- COCO `iscrowd` annotations are skipped.
- Mix day/night KAIST for better low-light performance.

After training, copy your `best.pt` to `foresight-beta/assets/models/` and point `scripts/yolo_detection.py` to the new weights.

## How It Works

### Main Control Panel
- Start/Stop phone capture
- Toggle SAR mode
- Monitor system status
- View console output

### Phone Mirroring
1. Click "Start Capture" in the main panel
2. A separate scrcpy window will open showing your phone screen
3. The window is automatically positioned at coordinates (400, 100) with size 350x600

### SAR Mode
1. Ensure phone capture is active
2. Toggle the SAR Mode switch
3. YOLO detection will analyze the phone mirror area
4. Detected objects will be highlighted and logged

### Window Grouping
- When you minimize the main Argus window, all related windows minimize
- When you restore the main window, all related windows restore
- Closing the main window terminates all processes

## Troubleshooting

### Scrcpy Issues
- Ensure scrcpy is installed and in your PATH
- Check that USB debugging is enabled on your device
- Try different USB cables or ports
- Verify device authorization

### YOLO Detection Issues
- Install required Python packages: `pip install -r requirements.txt`
- For better detection, download proper YOLO model files
- Check Python path in the main.js file