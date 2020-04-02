import numpy as np
import pickle
def create_nums(direc,number=100000,save = 10000):
  steps = 10
  try:
    with open(direc, 'rb') as f:
      data = pickle.load(f)
  except:
    data = {}
  prev = len(data.keys())
  base = np.random.rand(3,2)*2-1
  for i in range(prev,prev+number+1):
    traj = []
    movement = np.random.rand(steps,2)*2-1
    for k in range(steps):
      signal = base+movement[k]
      traj.append(signal)
    # copied = np.tile(base,(1000,1))
    # noise = np.random.rand(1000,1)*2-1
    # signal = copied+noise
    traj = np.stack(traj)
    data[i] = traj
    # print(traj.shape)
    if i %save ==0:
      with open(direc, 'wb') as f:
        pickle.dump(data,f)

if __name__ == "__main__":
  number = 200000
  direc = "data/triangles.pickle"
  save = 10000
  create_nums(direc,number,save)
  # base = np.random.rand(1,10)
  # for i in range(5):
  #   print(base+np.random.rand(1,1)*2-1)
  # base = np.random.rand(3,2)*2-1
  # movement = np.random.rand(1,2)*2-1
  # signal = base+movement
  # print(base)
  # print(movement)
  # print(signal)


  
  
