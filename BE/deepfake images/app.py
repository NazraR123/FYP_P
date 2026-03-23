from flask import Flask, request, jsonify, render_template_string
from ultralytics import YOLO
import numpy as np
import cv2
import os
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image
import torch
from flask_cors import CORS
import traceback
import warnings


warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['OUTPUT_FOLDER'] = './static/outputs'

model = YOLO('F:/Project/BE/deepfake images/best.pt')
model.cpu()

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def process_image(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (832, 832))
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_normalized = np.float32(rgb_img) / 255
        
        results = model(rgb_img)
        
        predictions = {}
        if hasattr(results[0], 'probs') and results[0].probs is not None:
            probabilities = results[0].probs.data.cpu().numpy()
            class_names = model.names
            predictions = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        
        # target_layers = [model.model.model[-2], model.model.model[-3], model.model.model[-4]]
        target_layers =[model.model.model[-3]]
        
        cam = EigenCAM(model, target_layers, task='cls')
        
        grayscale_cam = cam(rgb_img)[0, :, :]
        
        cam_image = show_cam_on_image(
            img_normalized,
            grayscale_cam,
            use_rgb=True
        )

        if predictions:
            max_conf_class = max(predictions.items(), key=lambda x: x[1])
            text = f"{max_conf_class[0]}: {max_conf_class[1]:.2f}"
            cam_image = cv2.putText(cam_image, text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return predictions, cam_image
    except Exception as e:
        print(f"Error during processing: {e}")
        traceback.print_exc()
        raise


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(input_path)
    print(f"Saved uploaded image to {input_path}")

    try:
        predictions, cam_image = process_image(input_path)

        output_filename = f"output_{file.filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, cam_image_bgr)

        response_text = ", ".join([f"{key} {value:.2f}" for key, value in predictions.items()])
        return jsonify({
            'result': response_text,
            'output_image': output_path
        })
    except Exception as e:
        print(f"Error during processing: {e}")
        traceback.print_exc()  
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)


