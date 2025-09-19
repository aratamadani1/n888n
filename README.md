# Face Sense Agent

This repository contains a lightweight Python module that detects human faces
and estimates their emotional expression ("sense of face") using a pre-trained
FER+ neural network model. The core functionality lives in
`facesense/agent.py`, which provides a `FaceSenseAgent` class and a
command-line interface for analysing both images and video streams.

## Features

- Haar-cascade face detection powered by OpenCV.
- Emotion recognition with the FER+ ONNX model and ONNX Runtime.
- Pure NumPy pre-processing utilities that can be unit tested without heavy
  runtime dependencies.
- Command-line interface for analysing single images or live video feeds.

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Analyse a still image and save the annotated result:

```bash
python -m facesense.agent --image path/to/photo.jpg --output annotated.jpg
```

Analyse frames coming from the default webcam:

```bash
python -m facesense.agent --video 0
```

The first run automatically downloads the FER+ ONNX model from the ONNX
Model Zoo. If the automatic download fails (for example, due to limited
connectivity), download the model manually from the
[FER+ model page](https://github.com/onnx/models/tree/main/vision/body_analysis/emotion_ferplus)
and place it at `facesense/models/emotion-ferplus-8.onnx`.

## Running Tests

The test suite focuses on the NumPy preprocessing utilities:

```bash
pytest
```

These tests do not require OpenCV or ONNX Runtime to be installed.
