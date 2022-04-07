from flask import Flask,request
import random

app = Flask(__name__)

def getSeverity(imageURL) :
  results = ['No DR','Mild','Moderate','Severe','Proliferative DR']
  return random.choice(results)

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