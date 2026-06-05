#pip install ultralytics opencv-python numpy

import cv2
import numpy as np
import urllib.request
from ultralytics import YOLO
import smtplib
from email.message import EmailMessage

# Email configuration
EMAIL_ADDRESS = 'miniprojectinterdisciplinary@gmail.com'
EMAIL_PASSWORD = 'egvx wzvq ygqc jxrb'  # Use App Password if using Gmail
TO_EMAIL = 'miniprojectinterdisciplinary@gmail.com'

alert_sent = False  # Keeps track if alert was already sent

# EMAIL FUNCTION
def send_email_alert(count):
    global alert_sent
    if alert_sent:
        return
    alert_sent = True

    msg = EmailMessage()
    msg['Subject'] = 'Crowd Alert Detected'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = TO_EMAIL
    msg.set_content(f'Alert! Crowd size exceeded. Detected people: {count}')

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print("Email sent successfully!")
    except Exception as e:
        print("Email Error:", e)

# ESP32-CAM stream URL
url = 'http://192.168.137.57/cam-mid.jpg'  # Replace with your ESP32 IP

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Replace with your custom model if needed

# MAIN LOOP
while True:
    try:
        # Fetch image from ESP32-CAM
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)
        frame = cv2.resize(frame, (640, 480))

        # Detect with YOLOv8
        results = model(frame)[0]
        count = 0

        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label == "person":
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show count
        cv2.putText(frame, f"People Count: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Alert logic
        if count > 5:
            send_email_alert(count)
        else:
            alert_sent = False  # Reset alert state

        cv2.imshow("YOLOv8 - ESP32-CAM Crowd Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print("Error:", e)

cv2.destroyAllWindows()
