from flask import Flask,request
from flask_cors import CORS
from albumentations.pytorch import ToTensorV2
from albumentations import Compose,Resize
import numpy as np
import torch
import urllib
import os
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torch import nn

app = Flask(__name__)
CORS(app)

# Transformations for Images
VAL_Transforms = Compose(
  [
    Resize(width=256, height=256),
    ToTensorV2()
  ]
)
# Labels for Result
Label_Dict = {
  "0": "No Diabetic Retinopathy",
  "1": "Mild Diabetic Retinopathy",
  "2": "Moderate Diabetic Retinopathy",
  "3": "Severe Diabetic Retinopathy",
  "4": "Proliferate Diabetic Retinopathy",
}

# Making Model Instance
NewModel__ = EfficientNet.from_pretrained("efficientnet-b3")
NewModel__._fc = nn.Linear(1536, 5)
NewModel__.to('cpu')

# Loading Trained Model
checkpoint = torch.load("chk.pth.tar", map_location='cpu')
NewModel__.load_state_dict(checkpoint["state_dict"])

def getSeverity(imageURL) :

  NewModel__.eval()
  with urllib.request.urlopen(imageURL) as url:
    with open('temp2.jpg', 'wb') as f:
      f.write(url.read())

  # Getting Image in the Right Format
  NewImage = np.array(Image.open("temp2.jpg"))
  NewImage = VAL_Transforms(image=NewImage)["image"]
  NewImage = NewImage.unsqueeze(0)
  NewImage = NewImage.to('cpu', dtype=torch.float)

  with torch.no_grad():
    scores1 = NewModel__(NewImage) # Scores for all classes

  _, preds1 = scores1.max(1)

  severity = str(preds1.detach().cpu().numpy()[0])
  return severity
  #results = ['No DR','Mild','Moderate','Severe','Proliferative DR']
  #return secrets.choice(results)

@app.errorhandler(404)
def invalidRoute(e):
  return 'Invalid route'

@app.route('/')
def home():
  return '<h1>Backend API for miniproject </h1>'


@app.route('/severity')
def severity():
  imageURL = request.args.get("imageURL")
  severity = getSeverity(imageURL)
  return {
    'success':True,
    'severity':severity
  }

@app.route('/labels')
def getLabels():
  return Label_Dict;