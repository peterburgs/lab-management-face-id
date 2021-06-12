from db import db


class UserModel(db.Model):
    __tablename__ = "users"

    id = db.Column(db.String, primary_key=True)
    email = db.Column(db.String)
    fullName = db.Column(db.String)
    isHidden = db.Column(db.Boolean)
    faces = db.relationship("FaceModel", lazy="dynamic", primaryjoin="and_(UserModel.id==FaceModel.user_id, "
                                                                     "FaceModel.isHidden==False)")

    def json(self):
        return {"id": self.id, "email": self.email, "fullName": self.fullName,
                "faceNum": len([face for face in self.faces.all()])}

    def __init__(self, _id, email, fullName):
        self.id = _id
        self.email = email
        self.fullName = fullName
        self.isHidden = False

    @classmethod
    def find_by_email(cls, email):
        return cls.query.filter_by(email=email, isHidden=False).first()

    @classmethod
    def find_by_id(cls, _id):
        return cls.query.filter_by(id=_id, isHidden=False).first()

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()
