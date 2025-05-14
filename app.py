# %%
import os

from flask import Flask, render_template, request, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from datetime import datetime

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config['SECRET_KEY'] = 'your_secret_key_here'


app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

 
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


users = {'admin': {'password': 'admin123'}}


breast_cancer_model = keras.models.load_model('models/breas.h5', compile=False)
cervical_cancer_model = keras.models.load_model('models/cerv.h5', compile=False)



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if users.get(username) and users[username]['password'] == password:
            user = User(username)
            login_user(user)
            return redirect(url_for('index'))
        else:
            return "Invalid credentials", 401
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    return render_template('login.html', timestamp=timestamp)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))



@app.route('/breast_cancer_prediction', methods=['GET', 'POST'])
@login_required
def breast_cancer_prediction():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400
        file = request.files['file']
        if file and allowed_file(file.filename):
            cl1=['benign', 'malignant', 'normal']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        file.save(filepath)
        
        image = Image.open(filepath)
        image = image.resize((150,150))  
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  

        prediction=np.argmax( breast_cancer_model.predict(image_array))
        return render_template('breast_cancer.html', filename=filename, result=cl1[prediction])
    return render_template('breast_cancer.html')

@app.route('/cervical_cancer_prediction', methods=['GET', 'POST'])
@login_required
def cervical_cancer_prediction():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400
        file = request.files['file']
        if file and allowed_file(file.filename):
            cl2=['cervic_Dyskeratotic', 'cervic_Koilocytotic', 'cervic_Metaplastic', 'cervic_Parabasal', 'cervic_Superficial-Intermediate']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

           
        image = Image.open(filepath)
        image = image.resize((150,150))  
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  

        prediction =np.argmax( cervical_cancer_model.predict(image_array))

        return render_template('cervical_cancer.html', filename=filename, result=cl2[prediction])

    return render_template('cervical_cancer.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
