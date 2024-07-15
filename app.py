from flask import Flask, request, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from PIL import Image
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///lost_and_found.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static'
app.config['SECRET_KEY'] = 'super secret key'

db = SQLAlchemy(app)

class ImageEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(150), nullable=False)
    caption = db.Column(db.String(300), nullable=False)

def create_tables():
    with app.app_context():
        db.create_all()

create_tables()

# Load external libraries for image processing
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    model_id = 'Salesforce/blip-image-captioning-base'
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)
except ImportError:
    raise ImportError("Make sure to install required modules: transformers, torch")
except Exception as e:
    app.logger.error(f"Failed to load the BLIP model: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            image = Image.open(file_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**inputs)
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            new_entry = ImageEntry(image_path=file_path, caption=caption)
            db.session.add(new_entry)
            db.session.commit()
            flash('Image uploaded and caption generated successfully!', 'success')
            return redirect(url_for('gallery'))
    return render_template('edit_entry.html', entry=None)

@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def edit_entry(id):
    entry = ImageEntry.query.get_or_404(id)
    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename != '':
            filename = secure_filename(request.files['file'].filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            request.files['file'].save(file_path)
            entry.image_path = file_path
        entry.caption = request.form.get('caption', entry.caption)
        db.session.commit()
        flash('Entry updated successfully!', 'success')
        return redirect(url_for('gallery'))
    return render_template('edit_entry.html', entry=entry)

@app.route('/gallery')
def gallery():
    entries = ImageEntry.query.all()
    return render_template('gallery.html', entries=entries)

if __name__ == '__main__':
    app.run(debug=True)
