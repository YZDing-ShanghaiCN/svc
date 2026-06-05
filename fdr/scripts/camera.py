from __future__ import annotations

from pathlib import Path

import cv2


def capture_photo(
    output_path: Path,
    camera_index: int = 0,
    window_name: str = "Press SPACE to capture, Q to quit",
) -> Path:
    """
    Capture one photo from the local camera.

    Press SPACE to capture and save the frame.
    Press q to cancel.
    """
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera (index={camera_index}).")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame from camera.")

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if not cv2.imwrite(str(output_path), frame):
                    raise RuntimeError(f"Failed to save image to: {output_path}")
                return output_path

            if key == ord("q"):
                raise RuntimeError("Camera capture canceled by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

