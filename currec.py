import numpy
import flask
import io
from PIL import Image
from keras.models import load_model
from keras import layers
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input

import tensorflow as tf
graph = tf.get_default_graph()


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
#model = None



#def build_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
global model
model = load_model("currency_detector_test8.h5")
      

def prepare_image(image):
    # if the image mode is not RGB, convert it
    img = image.resize((224, 224))
    x = img_to_array(img)
    x = numpy.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = x / 255
    return x


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            with graph.as_default():
                image = prepare_image(image)
                predict = model.predict(image)
                pred = numpy.argmax(predict, axis=1)
            if pred == 0 or pred == 1:
                pred = 100
            elif pred == 2 or pred == 3:
                pred = 10
            elif pred == 4 or pred == 5:
                pred = 2000
            elif pred == 6 or pred == 7:
                pred = 200
            elif pred == 8 or pred == 9:
                pred = 500
            elif pred == 10 or pred == 11:
                pred = 50
            prob = numpy.max(predict, axis=1)
            if (prob < 0.5) or pred == 12:
                prob = 0
                pred = 0
            data["predictions"] = []
            #print(prob)
            # loop over the results and add them to the list of
            # returned predictions
            
            r = {"label": pred, "probability": float(prob)}
            data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == "__main__":

    print(("* Loading Keras model and Flask starting server..."
              "please wait until server has fully started"))
    
    app.run(host="192.168.43.5")