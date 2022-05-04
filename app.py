####################################################
# Program : DGACaps API
# Author : Roshith Eshanka Ruhunuhewa
# Created : 23/03/2022
# Updated : 01/05/2022
####################################################

####################################################
# Imports
####################################################
import os
import numpy as np
import pandas as pd
import pickle
import secrets
from flask import Flask, request, render_template, redirect, url_for, jsonify, flash
from werkzeug.exceptions import RequestEntityTooLarge, MethodNotAllowed
from keras.models import load_model
from layers.capsuleLayer import CapsuleLayer
####################################################

####################################################
# Global Variables
####################################################
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
sentences_length = 50
len_letters = 40

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024
app.config['SENTENCE_LENGTH'] = 50
app.config['LEN_LETTER'] = 40
####################################################

####################################################
# Supporting Functions
####################################################
def char_replace(domain):

    domain = domain.replace("ç", "c")
    domain = domain.replace("Ç", "C")
    domain = domain.replace("à", "a")
    domain = domain.replace("Ä", "A")
    domain = domain.replace("ä", "a")
    domain = domain.replace("À", "A")
    domain = domain.replace("Â", "A")
    domain = domain.replace("â", "a")
    domain = domain.replace("é", "e")
    domain = domain.replace("è", "e")
    domain = domain.replace("É", "E")
    domain = domain.replace("È", "E")
    domain = domain.replace("Ë", "E")
    domain = domain.replace("ë", "e")
    domain = domain.replace("Ê", "E")
    domain = domain.replace("ê", "e")
    domain = domain.replace("û", "u")
    domain = domain.replace("Û", "U")
    domain = domain.replace("ü", "u")
    domain = domain.replace("Ü", "U")
    domain = domain.replace("ï", "i")
    domain = domain.replace("Ï", "I")
    domain = domain.replace("î", "i")
    domain = domain.replace("Î", "I")
    domain = domain.replace("Ô", "O")
    domain = domain.replace("ô", "o")
    domain = domain.replace("Ö", "O")
    domain = domain.replace("ö", "o")
    domain = domain.replace("Ù", "U")
    domain = domain.replace("ù", "u")
    domain = domain.replace("ÿ", "y")
    domain = domain.replace("æ", "ae")
    domain = domain.replace("_", " ")
    domain = domain.replace("\n", "")
    domain = domain.replace("\r", "")
    
    return domain

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_char_map():
    charmap = open('characterMap.pkl','rb')
    char_map = pickle.load(charmap)
    charmap.close()
    return char_map

def set_flag(i):
    temp = np.zeros(app.config['LEN_LETTER'])
    temp[i] = 1
    return list(temp)

def prepare_X(X):
    domain_list = []
    domain_truncs = [str(i)[0:app.config['SENTENCE_LENGTH']] for i in X]
    for i in domain_truncs:
        temp = [set_flag(app.Chars[j]) for j in str(i)]
        for k in range(0,app.config['SENTENCE_LENGTH'] - len(str(i))):
            temp.append(set_flag(app.Chars["END"]))
        domain_list.append(temp)

    return domain_list

def pred(domains, prediction):
    return_results = []
    index = 0
    dgaCount = 0
    benignCount = 0
    for i in prediction:
        if i[0] > i[1]:
            return_results.append([domains[index], "DGA"])
            dgaCount += 1
        else:
            return_results.append([domains[index], "Benign"])
            benignCount += 1
        index += 1
    return return_results, dgaCount, benignCount

####################################################

app.Model = load_model('capsDGA_model.h5',custom_objects={'CapsuleLayer': CapsuleLayer})
app.Chars = load_char_map()

####################################################
# API Functions
####################################################
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods = ['GET','POST'])
def upload_file():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                flash('No file part in the request', 'error')
                return redirect(request.url)
            else:
                file = request.files['file']
                if file.filename == '':
                    flash('No file is selected', 'error')
                    return redirect(request.url)
                if file and allowed_file(file.filename):
                    filename = 'dgasToClassify'
                    filename = app.config['UPLOAD_FOLDER']+'/'+filename
                    file.save(os.path.join(filename))
                    flash('Successfully Uploaded!', 'success')
                    return redirect(request.url)
                else:
                    flash('File type unsupported', 'error')
                    return redirect(request.url)
        except RequestEntityTooLarge:
            flash('File size too large', 'error')
            return redirect(request.url)        
    return redirect(url_for('home'))

@app.route('/predict', methods = ['GET','POST'])
def predict():
    try:
        dp = pd.read_csv(app.config['UPLOAD_FOLDER']+'/dgasToClassify', sep = ',')
        dp = pd.DataFrame(dp)
        df = dp["urls"]
        domains = df.values.tolist()
        #domains_to_test = ["mskqpaiq.biz", "google.com", "appleborderlackentrancedump.com"]
        app.total = len(domains)
        X_pred = prepare_X([char_replace(e) for e in domains])
        prediction = app.Model.predict(X_pred)
        app.domains, app.dgaCount, app.benignCount = pred(domains,prediction)
        resp = {
            'response' : 'successful'
        }
        resp = jsonify(resp)
    except Exception as e:
        resp = {
            'response' : str(e)
        }
        resp = jsonify(resp)
    return resp

@app.route('/report', methods = ['GET','POST'])
def view():
    resp = {
        'header1' : 'Domain',
        'header2' : 'DGA/Benign',
        'urls' : app.domains,
        'totalurls' : app.total,
        'dgaC' : app.dgaCount,
        'benignC' : app.benignCount
    }
    resp = jsonify(resp)
    return resp

# @app.errorhandler(MethodNotAllowed)
# def handle_exception(e):
#     flash(str(e))
#     resp = {
#         "code": e.code,
#         "name": e.name,
#         "description": e.description,
#     }
#     resp = jsonify(resp)
#     return resp
####################################################

if __name__== '__main__':
    app.run(debug=True)