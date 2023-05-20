from flask import Flask, render_template, request, send_file
from PIL import Image
import os
import pickle
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
HISTORY_FILE = 'history.pkl'
MODEL_DIR = 'model'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Set upload folder in Flask configuration

load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
model = tf.keras.models.load_model('model.h5', options=load_options)


def autocrop_image(image):
    grayscale_image = image.convert('L')
    width, height = grayscale_image.size
    average_pixel = sum(grayscale_image.getdata()) // (width * height)
    threshold = 10

    left = width
    top = height
    right = 0
    bottom = 0

    for y in range(height):
        for x in range(width):
            pixel_value = grayscale_image.getpixel((x, y))
            if pixel_value < average_pixel - threshold:
                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)
    cropped_image = image.crop((left, top, right + 1, bottom + 1))
    return cropped_image


def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(5, activation='sigmoid'))  # Adjust the number of output classes (defect types)

    return model


def predict_defect(image):
    defect_type = None
    cropped_image = autocrop_image(image)
    processed_image = cropped_image.resize((180, 180))

    img_array = tf.keras.preprocessing.image.img_to_array(processed_image)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = predictions[0][0]

    if score >= 0.5:
        label = "Defective"
    elif 0.65 < predictions[0][4] > 0.5:
        defect_type = "Glaucoma"
        label="Glaucoma"
    elif 0.8 < predictions[0][2] >= 0.65:
        defect_type = "Cataract"
        label = "Cataract"
    elif predictions[0][3] >= 0.8:
        defect_type = "Diabetic Retinopathy"
        label = "Diabetic Retinopathy"
    else:
        label = "Non-defective"

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cropped_image.jpg')
    cropped_image.convert('RGB').save(image_path)

    return label, score, defect_type


def save_to_history(image_path, result):
    history = load_history()
    history.append((image_path, result))
    with open(HISTORY_FILE, 'wb') as file:
        pickle.dump(history, file)


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'rb') as file:
            history = pickle.load(file)
        return history
    return []


def reset_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error='No image selected.')

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error='No image selected.')

        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            print("Image saved to:", image_path)  # Debug output
            image = Image.open(image_path)
            cropped_image = autocrop_image(image)
            cropped_image.save(image_path)
            result = predict_defect(cropped_image)
            save_to_history(image_path, result)
            return render_template('index.html', result=result, image_path=image_path)

    history = load_history()
    return render_template('index.html', history=history)


@app.route('/reset', methods=['POST'])
def reset():
    reset_history()
    return render_template('index.html')


@app.route('/uploads/<path:filename>', methods=['GET'])
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
