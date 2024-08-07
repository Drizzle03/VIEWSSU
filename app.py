from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# MobileNet SSD 설정
model_path = './'
prototxt_path = model_path + 'deploy.prototxt'
weights_path = model_path + 'mobilenet_iter_73000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)

def detect_people(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    people_count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # 신뢰도 임계값
            idx = int(detections[0, 0, i, 1])
            if idx == 15:  # 클래스 ID 15는 "person"을 나타냅니다.
                people_count += 1

    return people_count

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({"error": "Invalid image"}), 400
    
    people_count = detect_people(image)
    return jsonify({"people_count": people_count})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
