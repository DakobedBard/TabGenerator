from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os

from flask import flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from tabgen.tablature import Tab

#with open('spam_model.pkl', 'rb') as f:
    #model = pickle.load(f)

UPLOAD_FOLDER  = '/home/mddarr/galvanize/capstone/tab_generator/data/upload'
ALLOWED_EXTESIONS  = set(['wav'])

app = Flask(__name__,static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTESIONS

@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        #CHECK  if post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename =='':
            flash('No selected file')
            return redirect(request.url)


        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file',
                                    filename=filename))
    
    return ''' <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>'''
    
    #eturn render_template('upload.html')



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    
    #prediction = model.predict_proba([data['user_input']])
    return "hllo"


@app.route('/tab/<filename>')
def tablature(filename):

    ## Have this route render the the tablature template.  

    #t1 = Tab('data/audio/audio_mic/00_BN1-129-Eb_solo_mic.wav')

    #t1 = Tab('data/user_audio/newfilename.wav')

    #audio_dir = 'data/audio/audio_mic'
    audio_dir = 'data/audio/audio_mic/'
    path_name = os.path.join(audio_dir, filename)

    t1 = Tab(path_name)
    notes = t1.notes
    tablature_lines = ''


    for line in t1.lines:
        tablature_lines += line.line_html
        tablature_lines += '\n'
    
    #eturn tablature['t']

    name =filename.split('./')[-1].split('.')[0]
    tablature = {'t': tablature_lines, 'name': name, 'tab_notes' : notes}

    return render_template('tab.html', title='Home', tab=tablature)


@app.route('/')
def audio_table():
    
    
    audio_dir = 'data/audio/audio_mic/'
    audio_dir_files = os.listdir(audio_dir)

    #user_audio_dir = 'data/user_audio'
    #user_audio_files = os.listdir(user_audio_dir)

    return render_template('table.html', title = 'Home', filenames = audio_dir_files)







