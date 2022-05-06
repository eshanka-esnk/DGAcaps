####################################################
# Program : DGACaps API
# Author : Roshith Eshanka Ruhunuhewa
# Created : 23/03/2022
# Updated : 05/05/2022
####################################################

####################################################
# Imports
####################################################
import os
import numpy as np
import pandas as pd
import pickle
import secrets
from flask import Flask, request, render_template, redirect, url_for, send_file, flash
from werkzeug.exceptions import RequestEntityTooLarge, MethodNotAllowed
from fpdf import FPDF
from keras.models import load_model
from layers.capsuleLayer import CapsuleLayer
####################################################

####################################################
# Global Variables
####################################################
UPLOAD_FOLDER = 'uploads'
PDF_FOLDER = 'pdf'
ALLOWED_EXTENSIONS = {'csv'}
sentences_length = 50
len_letters = 40

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PDF_FOLDER'] = PDF_FOLDER
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
    return render_template('home.html')

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
                    return redirect(url_for('predict'))
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
        app.total = len(domains)
        X_pred = prepare_X([char_replace(e) for e in domains])
        prediction = app.Model.predict(X_pred)
        app.domains, app.dgaCount, app.benignCount = pred(domains,prediction)
        flash('Classification Successful', 'success')
        return redirect(url_for('view'))
    except Exception as e:
        flash(str(e), 'error')
        return redirect(url_for('home'))

@app.route('/report')
def view():
    try:
        table_data = app.domains
        total = app.total
        malCount = app.dgaCount
        benCount = app.benignCount
        return render_template('report.html',table_data=table_data , total=total, malCount=malCount, benCount=benCount)
    except AttributeError:
        flash('No domains to report', 'error')
        return redirect(url_for('home'))

@app.route('/pdf')    
def pdf():
    try:
        app.PDF = FPDF()
        app.PDF.add_page()
        app.PDF.set_font("Times", size = 15)
        line_height = app.PDF.font_size * 2.5
        col_width = app.PDF.epw / 2
        app.PDF.cell(200, 10, txt = "Report", ln = 1, align = 'C')
        app.PDF.cell(50, 10, txt = "Total Count: "+str(app.total), ln = 0, align = 'L')
        app.PDF.cell(50, 10, txt = "Malicious Count: "+str(app.dgaCount), ln = 0, align = 'L')
        app.PDF.cell(50, 10, txt = "Benign Count: "+str(app.benignCount), ln = 1, align = 'L')
        for row in app.domains:
            for column in row:
                app.PDF.multi_cell(col_width, line_height, column, border=1,
                    new_x="RIGHT", new_y="TOP", max_line_height=app.PDF.font_size)
            app.PDF.ln(line_height)
        app.PDF.cell(200, 30, txt = "DGACaps © Copyright 2022", align = 'C')
        app.PDF.output(app.config['PDF_FOLDER']+'/Report.pdf')
        flash('PDF exported', 'success')
        return send_file(app.config['PDF_FOLDER']+'/Report.pdf', as_attachment=True)
    except Exception as e:
        flash(str(e), 'error')
        return redirect(url_for('view'))

@app.errorhandler(MethodNotAllowed)
def handle_exception(e):
    flash(str(e.description), 'error')
    return redirect(url_for('home'))
####################################################

if __name__== '__main__':
    app.run(debug=True)