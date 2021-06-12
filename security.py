from flask import request
import functools
from google.oauth2 import id_token
from google.auth.transport import requests
from werkzeug.datastructures import ImmutableMultiDict

CLIENT_ID = '842135343547-0l11d5qs9q4jqchn7i45p75e5bf2jpqf.apps.googleusercontent.com'


def require_auth():
    def decorator(func):
        @functools.wraps(func)
        def secure_function(*args, **kwargs):
            token = None

            if 'Authorization' in request.headers:
                token = request.headers["Authorization"].split(" ")[1]

            if not token:
                return {"message": "A valid token is missing"}

            try:
                idinfo = id_token.verify_oauth2_token(token, requests.Request(), CLIENT_ID)
                http_args = request.args.to_dict()
                http_args['user'] = idinfo
                request.args = ImmutableMultiDict(http_args)
                return func(*args, **kwargs)
            except ValueError:
                return {"message": "Invalid token"}, 401

        return secure_function

    return decorator
