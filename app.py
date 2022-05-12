import pickle
import joblib
from flask import Flask, render_template
from flask import request
import numpy as np

# shutil.rmtree('flask_session')
model = joblib.load(open('rf.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_res():
    skw = float(request.form.get('skewness'))
    var = float(request.form.get('varience'))
    cur = float(request.form.get('curtosys'))
    ent = float(request.form.get('entropy'))

    result = model.predict(np.array([skw,var,cur,ent]).reshape(1,4))
    if result[0] == 1:
        result = "REAL"
    else:
        result = "FAKE"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)