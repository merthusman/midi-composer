# AI MIDI Composer

An AI-powered MIDI analysis and generation tool that learns from your MIDI files and creates new compositions following music theory principles.

## Features

- Modern PyQt6-based GUI interface
- MIDI file analysis (tempo, key, note patterns)
- AI-powered music generation using TensorFlow
- Music theory-compliant composition
- Real-time MIDI preview
- Batch processing capabilities

## Setup

1. Ensure you have Anaconda installed on your system
2. Create a new conda environment:
```bash
conda create -n midi-composer python=3.10
conda activate midi-composer
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python src/main.py
```

## Compilation

To create a standalone executable:

1. Ensure all requirements are installed
2. Run the build script:
```bash
python build.py
```
3. The executable will be created in the `dist` directory
4. You can distribute the executable file `MIDI_Composer.exe` to other Windows systems

Note: The first run of the executable might take longer as it initializes TensorFlow.

## Project Structure

- `src/`
  - `main.py` - Main application entry point
  - `gui/` - PyQt6 GUI components
  - `model/` - TensorFlow model and training logic
  - `midi/` - MIDI processing utilities
  - `utils/` - Helper functions and utilities
- `resources/` - Application resources and icons
- `build.py` - Executable build script

## Usage

1. Launch the application
2. Upload MIDI files through the interface
3. Train the model on your MIDI dataset
4. Adjust generation parameters
5. Generate new MIDI compositions

## Requirements

- Python 3.10 or higher
- CUDA-capable GPU (optional, for faster training)
- Minimum 8GB RAM recommended 