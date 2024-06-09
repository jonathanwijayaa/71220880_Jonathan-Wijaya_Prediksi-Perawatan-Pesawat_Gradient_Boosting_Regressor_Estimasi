from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'data'

# Variabel Global untuk menyimpan model, scaler, dan feature columns
model_ttf = None
scaler = None
feature_columns = []

# fungsi untuk membaca dan train model
# dari file upload
def train_model(file_path):
    global model_ttf, scaler, feature_columns

    # Membaca dataset dari file_upload
    data = pd.read_csv(file_path)
    
    # Data cleaning
    data = data.dropna()
    
    # feature columns
    feature_columns = ['cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)] + [f'av{i}' for i in range(1, 22)] + [f'sd{i}' for i in range(1, 22)]
    
    # Normalisasi
    scaler = StandardScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    
    # Data distribution
    X = data[feature_columns]
    y_ttf = data['ttf']
    
    # Membagi data menjadi training set dan testing set
    X_train, X_test, y_ttf_train, y_ttf_test = train_test_split(X, y_ttf, test_size=0.2, random_state=42)
    
    # Membangun TTF model estimasi menggunakan Gradient Boosting
    # Akurasi yang tinggi, dan kemampuan menangani berbagai jenis data
    model_ttf = GradientBoostingRegressor(random_state=42)
    model_ttf.fit(X_train, y_ttf_train)
    
    # Prediksi TTF untuk data testing
    y_ttf_pred = model_ttf.predict(X_test)
    
    # Evaluasi TTF Model
    mse = mean_squared_error(y_ttf_test, y_ttf_pred)
    rmse = np.sqrt(mse)
    return rmse

# Halaman utama untuk file upload dan input
@app.route('/', methods=['GET', 'POST'])
def index():
    rmse = None
    result = None

    if request.method == 'POST':
        # Upload reference file
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                rmse = train_model(file_path)

        # Manual data input
        if 'cycle' in request.form:
            global model_ttf, scaler, feature_columns
            if model_ttf is not None and scaler is not None:
                user_data = {col: [float(request.form.get(col, 0))] for col in feature_columns}
                user_input_df = pd.DataFrame(user_data)
                user_input_df[feature_columns] = scaler.transform(user_input_df[feature_columns])
                user_ttf_pred = model_ttf.predict(user_input_df)

                if user_ttf_pred[0] <= 100:
                    result = 'Pesawat harus mendapatkan perawatan!'
                else:
                    result = 'Pesawat masih dalam kondisi baik.'

    return render_template('interface.html', rmse=rmse, result=result)


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
