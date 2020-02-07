from flask import Flask, request, render_template
from werkzeug import secure_filename
from imageio import imread
from objloc import get_coords
import os

app = Flask(__name__)

@app.route('/', methods = ["GET", "POST"])
def upload_file():
    if request.method=='POST':
        f = request.files.get('file')
        if f is None:
        	return "No Image Uploaded!!!"
        f.save(secure_filename(f.filename))
        img = imread(secure_filename(f.filename))
        os.remove(secure_filename(f.filename))
        coords = get_coords(img)
        return render_template("index.html", after=True, coords=coords)
    return render_template("index.html", after=False)

@app.after_request
def add_header(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1,firefox=1'
    response.headers['Cache-Control'] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

if __name__ == "__main__":
    app.run(debug=True)
