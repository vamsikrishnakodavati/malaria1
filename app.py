from __future__ import print_function
import os
from flask import Flask, render_template
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

from get_prediction import *

basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
app.config['SECRET_KEY'] = 'vamsi'
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(basedir, 'uploads') # you'll need to create a folder named uploads


photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

class UploadForm(FlaskForm):   #create a form for image upload
    photo = FileField(validators=[FileAllowed(photos, 'Image only!'), FileRequired('File was empty!')])
    submit = SubmitField('Upload')

@app.route('/', methods=['GET', 'POST'])
def predict():
    #pred_image=None
    form = UploadForm() # calling the form to render on template
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = photos.url(filename)
        predict_result = get_img_predict("uploads/"+filename)   # post the image data to get_prediction.py to get prediction result
        pred_image = file_url
    else:
        predict_result = None
        pred_image=None
    return render_template('prediction.html', form=form,pred_image=pred_image, predict_result=predict_result)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
