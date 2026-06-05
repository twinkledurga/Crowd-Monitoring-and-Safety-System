import cv2
import urllib.request
import numpy as np
import smtplib
from email.message import EmailMessage

# ESP32-CAM stream URL
url = 'http://192.168.137.57/cam-lo.jpg'  # Replace with your ESP32 IP

# Load Haar cascade for faces (more reliable)
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Email configuration
EMAIL_ADDRESS = 'miniprojectinterdisciplinary@gmail.com'
EMAIL_PASSWORD = 'egvx wzvq ygqc jxrb'  # Use App Password if using Gmail
TO_EMAIL = 'miniprojectinterdisciplinary@gmail.com'

alert_sent = False  # Alert flag

def send_email_alert(count):
    global alert_sent
    if alert_sent:
        return
    alert_sent = True
    msg = EmailMessage()
    msg['Subject'] = "⚠️ Crowd Alert Detected"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = TO_EMAIL
    msg.set_content(f"Alert! Crowd size exceeded limit. Detected people: {count}")

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print("✅ Email alert sent!")
    except Exception as e:
        print("❌ Email failed:", e)

while True:
    try:
        # Get frame from ESP32-CAM
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)
        
        # Resize & convert
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces (tweak parameters for better detection)
        faces = cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=3, 
            minSize=(30, 30)
        )

        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        count = len(faces)
        cv2.putText(frame, f"People Count: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Alert logic
        if count > 5:
            send_email_alert(count)
        else:
            alert_sent = False  # Reset if crowd drops

        # Show the frame
        cv2.imshow("Crowd Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    except Exception as e:
        print("⚠️ Error:", e)

cv2.destroyAllWindows()
