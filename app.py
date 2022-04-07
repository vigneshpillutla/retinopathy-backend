from flask import Flask,request
from flask_cors import CORS
import random
import secrets

app = Flask(__name__)
CORS(app)

def getSeverity(imageURL) :
  results = ['No DR','Mild','Moderate','Severe','Proliferative DR']
  return secrets.choice(results)

@app.errorhandler(404)
def invalidRoute(e):
  return 'Invalid route'

@app.route('/')
def home():
  return '<h1>Backend API for miniproject </h1>'


@app.route('/severity')
def severity():
  imageURL = request.args.get("imageURL");
  severity = getSeverity(imageURL)
  return {
    'success':True,
    'severity':severity
  }