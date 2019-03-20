from statistics import mode

import cv2
from keras.models import load_model
import numpy as np

from emotion_detector.utils.datasets import get_labels
from emotion_detector.utils.inference import detect_faces
from emotion_detector.utils.inference import draw_text
from emotion_detector.utils.inference import draw_bounding_box
from emotion_detector.utils.inference import apply_offsets
from emotion_detector.utils.inference import load_detection_model
from emotion_detector.utils.preprocessor import preprocess_input


def draw_emoji(coords, source_image, emotion, scale=(1, 1)):
    x, y, w, h = coords
    x -= int(w * (scale[0] - 1) // 2)
    y -= int(h * (scale[0] - 1))
    w = int(w * scale[0])
    h = int(h * scale[1])
    overlay = cv2.imread('graphics/' + emotion + '.png', -1)
    try:
        overlay = cv2.resize(overlay, (h, w), interpolation=cv2.INTER_CUBIC)
    except:
        return source_image
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGBA)
    alpha_for_source_image = 1 - overlay[:, :, 3] // 255
    alpha_for_overlay = 1 - alpha_for_source_image
    for c in range(0, 3):
        source_image[y:y + h, x:x + w, c] = overlay[:, :, c] * alpha_for_overlay + source_image[y:y + h, x:x + w, c] * alpha_for_source_image
    return source_image


detection_model_path = 'emotion_detector/trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = 'emotion_detector/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')
# hyper-parameters for bounding boxes shape
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming
cv2.namedWindow('Emotions', cv2.WINDOW_NORMAL)
video_capture = cv2.VideoCapture(0)
enabled = 0
while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    if enabled:
        faces = detect_faces(face_detection, gray_image)

        for face_coordinates in faces:

            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, emotion_target_size)
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            if emotion_probability < 0.4 and emotion_text != 'sad':
                emotion_text = 'neutral'
            rgb_image = draw_emoji(face_coordinates, rgb_image, emotion_text, (1.2, 1.2))

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Emotions', bgr_image)
    key = cv2.waitKey(10)
    if key == 32:
        enabled = 1 - enabled
    if key == 27:
        break
cv2.destroyWindow('Emotions')