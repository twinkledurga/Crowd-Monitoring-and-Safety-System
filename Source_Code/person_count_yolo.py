import cv2
import numpy as np
import urllib.request
import smtplib
from email.message import EmailMessage

# ESP32-CAM stream URL
url = 'http://192.168.137.57/cam-mid.jpg'  # Replace with your ESP32 IP

# Load YOLO
modelConfig = 'yolov3.cfg'
modelWeights= 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfig,modelWeights)
layer_names = net.getUnconnectedOutLayersNames()

# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Email configuration
EMAIL_ADDRESS = 'miniprojectinterdisciplinary@gmail.com'
EMAIL_PASSWORD = 'egvx wzvq ygqc jxrb'  # Use App Password if using Gmail
TO_EMAIL = 'miniprojectinterdisciplinary@gmail.com'
alert_sent = False

def send_email_alert(count):
    global alert_sent
    if alert_sent:
        return
    alert_sent = True
    msg = EmailMessage()
    msg['Subject'] = "Crowd Alert Detected"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = TO_EMAIL
    msg.set_content(f"Alert! Crowd size exceeded. Detected people: {count}")

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print("Email sent!")
    except Exception as e:
        print("Email error:", e)

while True:
    try:
        # Get frame from ESP32-CAM
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)
        frame = cv2.resize(frame, (640, 480))

        # Convert image to blob
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(layer_names)

        boxes, confidences, class_ids = [], [], []

        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if classes[class_id] == "person" and confidence > 0.5:
                    center_x, center_y, w, h = (detection[0:4] * np.array([640, 480, 640, 480])).astype("int")
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        count = len(indexes)

        # Draw detections
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {int(confidence*100)}%", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show count
        cv2.putText(frame, f"People Count: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Send alert if needed
        if count > 5:
            send_email_alert(count)
        else:
            alert_sent = False  # reset

        # Display
        cv2.imshow("YOLO Crowd Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print("Error:", e)

cv2.destroyAllWindows()
