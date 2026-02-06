from flask import Flask, render_template, request
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

# مجلد لرفع الصور
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# تحميل النموذج المدرب
model = YOLO("best.pt")  # تأكد أن best.pt هو اسم الموديل النهائي

# الصفحة الرئيسية - رفع الصورة والتنبؤ
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None

    if request.method == 'POST':
        # الحصول على الصورة
        image = request.files['image']
        if image:
            # حفظ الصورة بشكل مؤقت
            filename = secure_filename(image.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            image.save(image_path)

            # تشغيل النموذج على الصورة
            results = model(image_path)
            label = results[0].names[int(results[0].probs.top1)]
            prediction = label.capitalize()

    return render_template("index.html", prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
