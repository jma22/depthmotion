import numpy as np
import pickle
def create_nums(direc,number=100000,save = 10000):
  try:
    with open(direc, 'rb') as f:
      data = pickle.load(f)
  except:
    data = {}
  prev = len(data.keys())
  
  for i in range(prev,prev+number+1):
    print(i)
    base = np.random.rand(1,64)
    copied = np.tile(base,(256,1))
    noise = np.random.rand(256,1)*2-1
    signal = copied+noise
    data[i] = signal
    if i %save ==0:
      with open(direc, 'wb') as f:
        pickle.dump(data,f)

if __name__ == "__main__":
  number = 100000
  direc = "data/trajectories.pickle"
  save = 10000
  create_nums(direc,number,save)
  
  
  
