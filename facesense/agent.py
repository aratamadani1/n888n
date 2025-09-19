"""Tools for detecting facial expressions using a FER+ ONNX model.

This module implements a simple *AI agent* that locates human faces in
an image and predicts the most likely emotion (or "sense of face") for
each detected face. The implementation uses OpenCV for face detection
and ONNX Runtime for loading the pre-trained FER+ neural network model.

The :class:`FaceSenseAgent` class is intentionally written so that it can
be imported without the heavy runtime dependencies installed. Libraries
like ``opencv-python`` and ``onnxruntime`` are only imported when they
are required at runtime, which keeps unit tests lightweight.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["FaceSenseAgent", "FaceAnalysis"]


FERPLUS_MODEL_URL = (
    "https://github.com/onnx/models/raw/main/vision/body_analysis/"
    "emotion_ferplus/model/emotion-ferplus-8.onnx"
)


@dataclass
class FaceAnalysis:
    """Container that stores the result of a single face analysis."""

    box: Tuple[int, int, int, int]
    label: str
    confidence: float
    probabilities: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "box": list(self.box),
            "label": self.label,
            "confidence": float(self.confidence),
            "probabilities": {k: float(v) for k, v in self.probabilities.items()},
        }


class FaceSenseAgent:
    """Detect human faces and predict their emotional expression.

    Parameters
    ----------
    model_path:
        Optional path to an ONNX FER+ model. If omitted, the official
        pre-trained FER+ model published by the ONNX Model Zoo will be
        downloaded on-demand the first time it is required.
    providers:
        Optional iterable of provider names passed to
        :class:`onnxruntime.InferenceSession`.
    min_face_size:
        Minimum size of detected faces. Smaller values increase
        sensitivity but may lead to more false positives.
    """

    EMOTIONS: Sequence[str] = (
        "neutral",
        "happiness",
        "surprise",
        "sadness",
        "anger",
        "disgust",
        "fear",
        "contempt",
    )
    # The FER+ model expects inputs normalised with these statistics.
    _MODEL_MEAN: float = 0.5076
    _MODEL_STD: float = 0.2556

    def __init__(
        self,
        *,
        model_path: Optional[Path] = None,
        providers: Optional[Iterable[str]] = None,
        min_face_size: Tuple[int, int] = (48, 48),
    ) -> None:
        self.model_path = Path(model_path) if model_path else self._default_model_path()
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.providers = tuple(providers) if providers else None
        self.min_face_size = tuple(min_face_size)
        self._session = None
        self._face_detector = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze_frame(self, frame: np.ndarray) -> List[FaceAnalysis]:
        """Detect faces and compute their emotional probabilities.

        Parameters
        ----------
        frame:
            A BGR image represented as a NumPy array (as used by OpenCV).
        """

        faces, grayscale = self._detect_faces(frame)
        results: List[FaceAnalysis] = []
        for (x, y, w, h) in faces:
            face_patch = grayscale[y : y + h, x : x + w]
            label, confidence, distribution = self._predict_emotion(face_patch)
            results.append(
                FaceAnalysis(
                    box=(int(x), int(y), int(w), int(h)),
                    label=label,
                    confidence=confidence,
                    probabilities={emotion: distribution[i] for i, emotion in enumerate(self.EMOTIONS)},
                )
            )
        return results

    def annotate_frame(self, frame: np.ndarray, analyses: Sequence[FaceAnalysis]) -> np.ndarray:
        """Return a copy of ``frame`` with the analysis results overlaid."""

        cv2 = self._require_cv2()
        annotated = frame.copy()
        for result in analyses:
            x, y, w, h = result.box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{result.label}: {result.confidence * 100:.1f}%"
            cv2.putText(
                annotated,
                label,
                (x, max(y - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def analyze_image_file(
        self,
        image_path: Path,
        *,
        display: bool = False,
        save_path: Optional[Path] = None,
    ) -> List[FaceAnalysis]:
        """Analyse a still image located at ``image_path``."""

        cv2 = self._require_cv2()
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        results = self.analyze_frame(image)
        annotated = self.annotate_frame(image, results)

        if save_path:
            cv2.imwrite(str(save_path), annotated)
        if display:
            cv2.imshow("FaceSense", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return results

    def analyze_video_stream(
        self,
        source: int | str = 0,
        *,
        display: bool = True,
        stop_key: str = "q",
        throttle: Optional[float] = None,
    ) -> Iterable[List[FaceAnalysis]]:
        """Analyse frames from a webcam or video file.

        Parameters
        ----------
        source:
            Index of the camera to open or a path to a video file.
        display:
            If ``True`` (default), annotated frames are shown in a window.
        stop_key:
            Keyboard key to stop the stream when in display mode.
        throttle:
            Optional delay (in seconds) between consecutive frames. This
            can be useful when analysing pre-recorded clips to slow down
            playback for inspection.
        """

        cv2 = self._require_cv2()
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        try:
            while True:
                ret, frame = capture.read()
                if not ret:
                    break
                analyses = self.analyze_frame(frame)

                if display:
                    annotated = self.annotate_frame(frame, analyses)
                    cv2.imshow("FaceSense", annotated)
                    if cv2.waitKey(1) & 0xFF == ord(stop_key):
                        break
                if throttle:
                    time.sleep(throttle)
                yield analyses
        finally:
            capture.release()
            if display:
                cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Implementation helpers
    # ------------------------------------------------------------------
    def _detect_faces(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cv2 = self._require_cv2()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = self._get_face_detector(cv2)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=self.min_face_size,
        )
        return faces, gray

    def _get_face_detector(self, cv2_module):
        if self._face_detector is None:
            cascade_path = Path(cv2_module.data.haarcascades) / "haarcascade_frontalface_default.xml"
            detector = cv2_module.CascadeClassifier(str(cascade_path))
            if detector.empty():
                raise RuntimeError("Failed to load Haar cascade for face detection.")
            self._face_detector = detector
        return self._face_detector

    def _predict_emotion(self, face_patch: np.ndarray) -> Tuple[str, float, np.ndarray]:
        session = self._get_session()
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_blob = self.preprocess_face(face_patch)
        raw_output = session.run([output_name], {input_name: input_blob})[0][0]
        probabilities = self.softmax(raw_output)
        best_idx = int(np.argmax(probabilities))
        return self.EMOTIONS[best_idx], float(probabilities[best_idx]), probabilities

    def _get_session(self):
        if self._session is None:
            self._ensure_model()
            import onnxruntime as ort  # Imported lazily.

            providers = list(self.providers) if self.providers else None
            self._session = ort.InferenceSession(str(self.model_path), providers=providers)
        return self._session

    def _ensure_model(self) -> None:
        if self.model_path.exists():
            return
        try:
            import urllib.request

            with urllib.request.urlopen(FERPLUS_MODEL_URL) as response:
                data = response.read()
        except Exception as exc:  # pragma: no cover - network failure handling
            raise RuntimeError(
                "Unable to download FER+ model automatically. Please download "
                f"it manually from {FERPLUS_MODEL_URL} and place it at {self.model_path}."
            ) from exc

        self.model_path.write_bytes(data)

    # ------------------------------------------------------------------
    # Static helpers utilised by both runtime logic and unit tests
    # ------------------------------------------------------------------
    @staticmethod
    def preprocess_face(face_patch: np.ndarray) -> np.ndarray:
        """Prepare a face crop for consumption by the FER+ model."""

        if face_patch.ndim == 3:
            face_patch = FaceSenseAgent.rgb_to_grayscale(face_patch)
        face_patch = FaceSenseAgent._resize_to_square(face_patch, 64)
        face_patch = face_patch.astype(np.float32) / 255.0
        face_patch = (face_patch - FaceSenseAgent._MODEL_MEAN) / FaceSenseAgent._MODEL_STD
        return face_patch[np.newaxis, np.newaxis, :, :]

    @staticmethod
    def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Expected an RGB image with shape (H, W, 3)")
        # Use ITU-R BT.601 conversion which is also used by OpenCV.
        coefficients = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return np.tensordot(image[..., :3], coefficients, axes=([-1], [0]))

    @staticmethod
    def _resize_to_square(image: np.ndarray, size: int) -> np.ndarray:
        if image.ndim != 2:
            raise ValueError("_resize_to_square expects a 2-D grayscale image")
        src_h, src_w = image.shape
        if src_h == size and src_w == size:
            return image.astype(np.float32)

        # Prepare floating point arrays for interpolation.
        image = image.astype(np.float32)
        y_coords = np.linspace(0, src_h - 1, size)
        x_coords = np.linspace(0, src_w - 1, size)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        x0 = np.floor(x_grid).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, src_w - 1)
        y0 = np.floor(y_grid).astype(np.int32)
        y1 = np.clip(y0 + 1, 0, src_h - 1)

        Ia = image[y0, x0]
        Ib = image[y1, x0]
        Ic = image[y0, x1]
        Id = image[y1, x1]

        x1_minus_x0 = np.maximum(x1 - x0, 1)
        y1_minus_y0 = np.maximum(y1 - y0, 1)

        wx = (x_grid - x0) / x1_minus_x0
        wy = (y_grid - y0) / y1_minus_y0

        wa = (1 - wx) * (1 - wy)
        wb = (1 - wx) * wy
        wc = wx * (1 - wy)
        wd = wx * wy

        resized = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return resized.astype(np.float32)

    @staticmethod
    def softmax(logits: Sequence[float]) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float32)
        logits = logits - np.max(logits)
        exps = np.exp(logits)
        return exps / np.sum(exps)

    @staticmethod
    def _default_model_path() -> Path:
        return Path(__file__).resolve().parent / "models" / "emotion-ferplus-8.onnx"

    @staticmethod
    def _require_cv2():
        try:
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "OpenCV (cv2) is required for face detection but is not installed."
            ) from exc
        return cv2


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Detect the sense of human faces in images or video streams.")
    parser.add_argument("--image", type=Path, help="Path to an image to analyse")
    parser.add_argument("--video", help="Camera index or video file to analyse")
    parser.add_argument("--output", type=Path, help="Optional path to save an annotated image")
    parser.add_argument("--no-display", action="store_true", help="Do not display annotated output")
    parser.add_argument("--json", type=Path, help="Write raw JSON analysis to this file")
    parser.add_argument("--providers", nargs="*", help="ONNX Runtime execution providers to use")
    parser.add_argument("--model", type=Path, help="Custom path to an ONNX FER+ model")

    args = parser.parse_args(argv)

    if bool(args.image) == bool(args.video):
        parser.error("Please provide exactly one of --image or --video")

    agent = FaceSenseAgent(model_path=args.model, providers=args.providers)

    if args.image:
        analyses = agent.analyze_image_file(args.image, display=not args.no_display, save_path=args.output)
        if args.json:
            args.json.write_text(json.dumps([analysis.to_dict() for analysis in analyses], indent=2))
    else:
        display = not args.no_display
        try:
            for analyses in agent.analyze_video_stream(args.video, display=display):
                if args.json:
                    args.json.write_text(json.dumps([analysis.to_dict() for analysis in analyses], indent=2))
        except KeyboardInterrupt:
            pass
    return 0


def main() -> None:
    sys.exit(_cli())


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    main()
