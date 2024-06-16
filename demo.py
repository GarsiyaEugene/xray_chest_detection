from ultralytics import YOLO
import cv2
from flask import Flask, render_template, request
import math


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
img_path = f"{app.config['UPLOAD_FOLDER']}/uploaded.png"

@app.route('/detection', methods=['POST'])
def detect():
    file = request.files['file']
    file.save(img_path)

    results_multi = inference_models[0](f"{app.config['UPLOAD_FOLDER']}/uploaded.png", save=False, show=False, imgsz=imgsz, conf=0.2, iou=0.6, half=False, device='cpu', agnostic_nms=True)
    results_binary = inference_models[1](f"{app.config['UPLOAD_FOLDER']}/uploaded.png", save=False, show=False, imgsz=imgsz, conf=0.2, iou=0.5, half=False, device='cpu')
    results_1 = inference_models[2](f"{app.config['UPLOAD_FOLDER']}/uploaded.png", save=False, show=False, imgsz=imgsz, conf=0.2, iou=0.5, half=False, device='cpu')
    results_4 = inference_models[3](f"{app.config['UPLOAD_FOLDER']}/uploaded.png", save=False, show=False, imgsz=imgsz, conf=0.03, iou=0.5, half=False, device='cpu')

    r_i_to_title = {0: 'Multi',1: 'Binary',2: 'Cardiomegaly only',3: 'Mass/Nodule only'}
    for r_i, r in enumerate([results_multi[0], results_binary[0], results_1[0], results_4[0]]):
        plotted_pred = r.plot()

        FONT_SCALE = 2e-3
        THICKNESS_SCALE = 1e-3
        font_scale = min(plotted_pred.shape[1], plotted_pred.shape[0]) * FONT_SCALE
        thickness = math.ceil(min(plotted_pred.shape[1], plotted_pred.shape[0]) * THICKNESS_SCALE)

        (w, h), _ = cv2.getTextSize(r_i_to_title[r_i], cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        plotted_pred = cv2.rectangle(plotted_pred, (int(plotted_pred.shape[1]/2-w/2), int(h/2)), (int(plotted_pred.shape[1]/2+w/2), int(1.5*h)), (255, 0, 0), -1)

        plotted_pred = cv2.putText(plotted_pred, r_i_to_title[r_i], (int(plotted_pred.shape[1]/2-w/2), int(1.5*h) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        if r_i==0:
            img = plotted_pred.copy()
        else:
            img = cv2.hconcat((img, plotted_pred))

    cv2.imwrite(img_path, img)

    return render_template("index_post.html", value=img_path)

@app.route("/")
def form():
    return render_template("index.html")

if __name__ == "__main__":
    imgsz = 800
    inference_models = []
    for model_option in ['corrected_multi_800', 'corrected_binary_800', 'corrected_binary_class_1_800', 'corrected_binary_class_4_800']:
        model = YOLO(f"checkpoints/{model_option}.pt")
        inference_models.append(model)

    app.run(host="127.0.0.1", port=8080, debug=True)




#


