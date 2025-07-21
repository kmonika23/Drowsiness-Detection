import winsound
from flask import Flask, render_template, Response
import cv2
from transformers import AutoImageProcessor, FocalNetForImageClassification
from PIL import Image
import torch
import numpy as np
import time

app = Flask(__name__)

# Load Hugging Face model
processor = AutoImageProcessor.from_pretrained("MichalMlodawski/open-closed-eye-classification-focalnet-base")
model = FocalNetForImageClassification.from_pretrained("MichalMlodawski/open-closed-eye-classification-focalnet-base")

cap = cv2.VideoCapture(0)

closed_eyes_frame_count = 0
DROWSINESS_THRESHOLD = 20  # Number of consecutive frames

def predict_eye_state(eye_img):
    # Convert image to PIL format
    eye_pil = Image.fromarray(cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB))
    inputs = processor(images=eye_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(-1).item()
    return "Open" if predicted_class == 1 else "Closed"

def gen_frames():
    global closed_eyes_frame_count
    while True:
        success, frame = cap.read()
        if not success:
            break

        # For demo purposes, we extract a fixed region (simulate eye region)
        h, w = frame.shape[:2]
        eye_img = frame[h//4:h//2, w//4:w*3//4]

        state = predict_eye_state(eye_img)

        if state == "Closed":
            closed_eyes_frame_count += 1
        else:
            closed_eyes_frame_count = 0

        # Overlay text
        label = f"Eye: {state}"
        if closed_eyes_frame_count >= DROWSINESS_THRESHOLD:
            label = "DROWSINESS DETECTED!"
            cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
        else:
            cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
