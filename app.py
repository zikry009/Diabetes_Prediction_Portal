from flask import *
from sklearn.externals import joblib
import json
import os
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict',methods=["post"])
def predict():
    formvalues = request.form
    path1 = "/static/json/"
    with open(os.path.join(os.getcwd()+"/"+path1,'file.json'), 'w') as f:
        json.dump(formvalues, f)
    with open(os.path.join(os.getcwd()+"/"+path1,'file.json'), 'r') as f:
        values = json.load(f)
    df = pd.DataFrame(json_normalize(values))
    model_path=os.getcwd()+"/static/model/diabetes.pkl"
    model = joblib.load(model_path)
    result = model.predict(df)
    a=np.array(1)
    if result.astype('int')==a.astype('int'):
        msg="Success"
    else:
        msg = "Unsuccess"
    positive_percent= model.predict_proba(df)[0][1]*100
    return render_template("index.html",msg=msg,prob=positive_percent,**request.args)



if __name__ == '__main__':
    app.run()
