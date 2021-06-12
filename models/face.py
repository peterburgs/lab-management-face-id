from db import db


class FaceModel(db.Model):
    __tablename__ = "faces"

    id = db.Column(db.Integer, primary_key=True)
    embedding = db.Column(db.String)
    imageUrl = db.Column(db.String)
    isHidden = db.Column(db.Boolean)
    user_id = db.Column(db.String, db.ForeignKey('users.id'))
    user = db.relationship('UserModel')

    def __init__(self, embedding, user_id, imageUrl):
        self.embedding = embedding
        self.user_id = user_id
        self.imageUrl = imageUrl
        self.isHidden = False

    def json(self):
        return {"user": self.user_id, "embedding": self.embedding, "imageUrl": self.imageUrl, "isHidden": self.isHidden}

    @classmethod
    def find_by_user_id(cls, user_id):
        return cls.query.filter_by(user_id=user_id, isHidden=False).all()

    @classmethod
    def delete_by_user_id(cls, user_id):
        cls.query.filter_by(user_id=user_id).update(dict(isHidden=True))
        db.session.commit()

    @classmethod
    def get_all(cls):
        return cls.query.filter_by(isHidden=False).all()

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()
