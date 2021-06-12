import os
import time

from flask import Flask
from flask_restful import Api, Resource, reqparse
from flask import request

from models.face import FaceModel
from models.user import UserModel
from security import require_auth
from sklearn.metrics import accuracy_score
import werkzeug
import cv2
import numpy as np
import insightface
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from flask_cors import CORS
from werkzeug.utils import secure_filename
import copy

UPLOAD_FOLDER = './assets/uploads/'

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['PROPAGATE_EXCEPTIONS'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("SQLALCHEMY_DATABASE_URI", "sqlite:///data.db")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
api = Api(app)


@app.before_first_request
def create_tables():
    db.create_all()


class User(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('id', type=str, required=True, help="ID cannot be left blank!")
    parser.add_argument('email', type=str, required=True, help="Email cannot be left blank!")
    parser.add_argument('fullName', type=str, required=True, help="Full name cannot be left blank!")

    @require_auth()
    def post(self):
        request_data = User.parser.parse_args()

        if UserModel.find_by_email(request_data['email']):
            return {'message': "A user with that email already exists"}, 400

        user = UserModel(request_data["id"], request_data['email'], request_data['fullName'])

        user.save_to_db()

        return {'message': 'User created successfully'}, 201

    @require_auth()
    def get(self):
        user = UserModel.find_by_email(request.args['user']['email'])
        if user:
            return user.json()
        return {'message': 'User not found'}, 404


def plot_bbox(plt, ymin, xmin, ymax, xmax, img_height=1, img_width=1, boxcolor='b', boxedgewidth=3, label_str='',
              fontsize=8, text_background='y'):
    (ymin, ymax) = (int(y * img_height) for y in (ymin, ymax))  # convert to absolute coordinates
    (xmin, xmax) = (int(x * img_width) for x in (xmin, xmax))
    import matplotlib.patches as patches
    rect = patches.Rectangle((xmin, ymin), width=xmax - xmin, height=ymax - ymin, linewidth=boxedgewidth,
                             edgecolor=boxcolor, facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)
    plt.text(xmin, ymin, label_str, fontsize=fontsize, backgroundcolor=text_background)


def parseImage(image_file):
    image = np.asarray(bytearray(image_file.read()), dtype="uint8")
    return cv2.imdecode(image, cv2.IMREAD_COLOR)


def face2Embedding(image):
    model = insightface.app.FaceAnalysis(name="antelope", root="insight-face")
    ctx_id = -1
    model.prepare(ctx_id=ctx_id)
    faces = model.get(image)
    if len(faces) < 1:
        return None, "Cannot detect face"
    if len(faces) > 1:
        return None, "Too many faces"
    return faces[0].embedding, faces[0].bbox.astype(np.int).flatten()


class VerifyFace(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('image', type=werkzeug.datastructures.FileStorage, location='files', required=True,
                        help="Image file is required")

    @require_auth()
    def post(self):
        args = VerifyFace.parser.parse_args()
        image_file = args['image']

        image = parseImage(image_file)
        result = face2Embedding(image)

        if result[0] is not None:
            try:
                embedding, bbox = result
                # Print face detection info
                print("\tembedding:%s" % embedding)

                # Plot bounding box
                user = UserModel.find_by_email(request.args['user']['email'])
                plt.imshow(image)
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                plt.imshow(img)
                plot_bbox(plt, bbox[1], bbox[0], bbox[3], bbox[2], label_str=user.fullName)
                plt.show()

                embedding = np.reshape(embedding, (1, 512))

                faces = FaceModel.get_all()
                X = []
                y = []
                labels = []
                for i in range(len(faces)):
                    faces[i].embedding = faces[i].embedding.replace('[', '')
                    faces[i].embedding = faces[i].embedding.replace(']', '')

                    user_embedding = np.fromstring(faces[i].embedding, sep=",")
                    X.append(user_embedding)
                    y.append(faces[i].user_id)
                    if faces[i].user_id not in labels:
                        labels.append(faces[i].user_id)

                X = np.array(X)
                y = np.array(y)
                in_encoder = Normalizer(norm='l2')
                X = in_encoder.transform(X)
                out_encoder = LabelEncoder()
                out_encoder.fit(y)
                y = out_encoder.transform(y)
                model = SVC(kernel='linear', probability=True)
                model.fit(X, y)

                embedding = in_encoder.transform(embedding)
                y_pred = model.predict(embedding)
                y_pred_proba = model.predict_proba(embedding)

                if y_pred_proba[0, y_pred[0]] < 0.75:
                    return {"message": "Unknown face"}, 401

                user_id = out_encoder.inverse_transform(y_pred)[0]
                found_user = UserModel.find_by_id(user_id)
                if found_user is not None and found_user.id == user.id:
                    return {"user": found_user.json(), "status": 200}
                else:
                    return {"message": "Wrong Face ID", "status": 401}, 401
            except ValueError:
                return {"message": "Something went wrong", "status": 500}, 500
        return {'message': result[1], "status": 400}, 400


class Face(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('image1', type=werkzeug.datastructures.FileStorage, location='files', required=True,
                        help="Image 1 file is required")
    parser.add_argument('image2', type=werkzeug.datastructures.FileStorage, location='files', required=True,
                        help="Image 2 file is required")

    @require_auth()
    def post(self):
        args = Face.parser.parse_args()
        image_file = args['image1']
        image_file_to_save = args['image2']

        user = UserModel.find_by_email(request.args['user']['email'])

        if user:
            filename = secure_filename(image_file_to_save.filename)
            ext = filename.split(".")
            filename = f"{user.id}_{round(time.time())}.{ext[1]}"
            image_file_to_save.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            try:
                image = parseImage(image_file)
                result = face2Embedding(image)
                if result[0] is not None:
                    embedding, bbox = result

                    # Plot bounding box
                    plt.imshow(image)
                    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    plt.imshow(img)
                    plot_bbox(plt, bbox[1], bbox[0], bbox[3], bbox[2], label_str=user.fullName)
                    plt.show()

                    embedding = np.array2string(embedding, separator=',')

                    face = FaceModel(embedding, user.id, f"/assets/uploads/{filename}")
                    face.save_to_db()
                    faces = FaceModel.find_by_user_id(user.id)
                    return {"faceNum": len(faces), "status": 201}, 201
                else:
                    return {'message': result[1], "status": 400}, 400
            except ValueError:
                return {"message": "An error occur when inserting the face.", "status": 500}, 500
        return {'message': 'Cannot find user', "status": 400}, 400


class DeleteFace(Resource):
    @require_auth()
    def delete(self, user_id):
        try:
            user = UserModel.find_by_id(user_id)
            if user:
                try:
                    FaceModel.delete_by_user_id(user.id)
                    return {"message": "Delete faces successfully", "status": 200}, 200
                except ValueError:
                    return {"message": "An error occur when deleting faces.", "status": 500}, 500
        except ValueError:
            return {'message': 'Cannot find user', "status": 400}, 400


api.add_resource(Face, "/face")
api.add_resource(VerifyFace, "/verify")
api.add_resource(User, "/user")
api.add_resource(DeleteFace, "/face/<string:user_id>")

if __name__ == '__main__':
    from db import db

    db.init_app(app)
    app.run(port=5000, debug=True)
