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

  severity = Label_Dict[str(preds1.detach().cpu().numpy()[0])]
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
  # imageURL = "data:image/jpeg;base64,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"
  severity = getSeverity(imageURL)
  return {
    'success':True,
    'severity':severity
  }

@app.route('/labels')
def getLabels():
  return Label_Dict;