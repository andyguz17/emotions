import sys

sys.path.append('./helpers')

import grid
import pandas as pd

mFrames = grid.meshFrame()

angry = mFrames.setData('./images/angry', 0)
fear = mFrames.setData('./images/fear', 1)
happy = mFrames.setData('./images/happy', 2)
neutral = mFrames.setData('./images/neutral', 3)
sad = mFrames.setData('./images/sad', 4)
surprise = mFrames.setData('./images/surprise', 5)
disgust = mFrames.setData('./images/disgust', 6)

data = pd.concat([angry, fear, happy, neutral, sad, surprise])
data.to_csv('../data.csv')