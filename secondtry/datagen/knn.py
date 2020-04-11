import pickle
import numpy as np
import math


data_path = '../data/cube_data.pickle'

with open(data_path,"rb") as f:
  data = pickle.load(f)

print(data.shape)