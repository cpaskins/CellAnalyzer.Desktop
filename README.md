# CellAnalyzer Desktop

CellAnalyzer Desktop is a Windows desktop application for quantitative cell culture image analysis.  
It combines a modern C# (WPF) user interface with a Python-based image processing engine to deliver fast, reproducible cell counting and area metrics.

The project is designed to showcase:
- Clean C# desktop application architecture
- Python-based scientific image processing
- Cross-language integration
- Practical, real-world engineering decisions

---

## Features

- Load microscopy images and perform automated cell detection
- Adjustable analysis parameters (thresholds, morphology, scaling, etc.)
- Real-time visualization of:
  - Original image
  - Processed overlay
- Quantitative outputs:
  - Cell count
  - Total contour area
  - Mean contour area
- Save and load parameter presets
- Dark-mode modern UI
- Python engine packaged as a standalone executable

---

## Architecture Overview

The application is split into two layers:

### Desktop UI (C# / WPF)
- Handles user interaction and visualization
- Collects analysis parameters
- Displays images and metrics
- Manages presets and workflows

### Analysis Engine (Python)
- Performs all image processing and analytics
- Uses OpenCV and NumPy
- Exposes a CLI interface
- Packaged into a standalone executable using PyInstaller

Communication between the UI and engine happens via:
- JSON parameter files
- JSON result files
- Image outputs (overlay, mask)

This separation keeps the UI responsive and the analysis code reusable.

---

## Project Structure
<img src="BlockDiagram.svg" alt="CellAnalyzer System" width="1200" />
```text
CellAnalyzer.Desktop/
│
├─ CellAnalyzer.Desktop/        # C# WPF application
│  ├─ MainWindow.xaml
│  ├─ Services/
│  │  └─ PythonRunner.cs
│  ├─ Models/
│  │  ├─ AnalysisParameters.cs
│  │  └─ ParameterPreset.cs
│  └─ Themes/
│     └─ DarkTheme.xaml
│
├─ Python/                      # Python analysis engine
│  ├─ cli.py
│  ├─ masterFunction.py
│  ├─ Processing/
│  ├─ parameters.py
│  └─ venv/                     # Not tracked in Git
│
└─ README.md
