import os
import pickle
import timeit
from random import randint

import numpy
import numpy as np
import cupy as cp
import sys

from data_prep.data_utils import get_veg_data, get_veg_training_data, roll_data_channel_last
from data_prep.lbl_utils import get_sorted_veg_from_excel
#from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check_gpu import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers_gpu import *
from cs231n.fast_layers_gpu import *
from cs231n.solver import Solver
from cs231n.solver_gpu import Solver as Solver_gpu

#data = get_CIFAR10_data()

from cs231n.classifiers.convnet_gpu import ConvNet as ConvNet_gpu

from cs231n.classifiers.convnet import ConvNet
from cs231n.classifiers.convnet_kah import ConvNet as ConvNet_kah

class Tee(object):
  '''
  Writes console to file too.
  '''
  def __init__(self, *files):
    self.files = files
  
  def write(self, obj):
    for f in self.files:
      f.write(obj)
      f.flush()  # If you want the output to be visible immediately
  
  def flush(self):
    for f in self.files:
      try:
        f.flush()
      except:
        pass

def cpu_solver(datav):
  with open('./logs/.last-cpu.txt', 'r+') as fl:
    tmp = fl.read().replace('\n', '')
    os.mkdir("./logs/cpu-" + tmp)
    
    log = open("./logs/cpu-"+tmp+"/"+tmp+"-log-cpu.txt", "w")
    fl.seek(0)
    fl.write(str(int(tmp) + 1))
    fl.truncate()
    
    nameCk = "./logs/cpu-"+tmp+"/"+tmp+"-run-cpu"

  original = sys.stdout
  sys.stdout = Tee(sys.stdout, log)
  
  for i in range(1):
    start_time = timeit.default_timer()
    #.001, .000001
    #lr = np.random.uniform(1e-2, 1e-5)
    lr = 1e-4
    # Make sure your input_dim are same order as the shape of your images.  As seen the color channels are last for me.
    # Switching them around leads to errors
    hidden_dim = 512
    print(hidden_dim)
    #Must be odd number
    filter_size = 5
    
    model = ConvNet_kah(weight_scale=0.01, hidden_dim=hidden_dim, reg=0.001, filter_size=filter_size, num_filters=(7, 7, 7, 7),
                    num_classes=int(datav['y_train'].max()+1), input_dim=(3,46,46),dtype=np.float32)
    
    print("lr: %e" % (lr))
    
    batch_size = 100
    epochs=20
    #, "beta1" : 0.99, "beta2":0.9999
    solver = Solver(model, datav,
                        num_epochs=epochs, batch_size=batch_size,
                        update_rule="adam",
                        optim_config={"learning_rate": lr},
                        checkpoint_name=nameCk,
                        verbose=True, print_every=50,
                        num_val_samples=300, num_train_samples=400)
    
    solver.train()
    acc = solver.check_accuracy(solver.X_val, solver.y_val)
    log.write("lr: %e, acc: %f\n" % (lr, acc))
    print(">> lr: %e, acc: %f\n" % (lr, acc))
    elapsed = timeit.default_timer()
    print("<**> " + str(elapsed))
    log.write("Time: " + str(elapsed))
    log.write("Lr: "+str(lr))
    log.write("Batch_Size: "+str(batch_size))
    log.write("Epochs: "+str(epochs))
    log.write("Hidden_Dim: "+str(hidden_dim))
    log.write("Filter_Size: "+str(filter_size))
  
  log.close()
  
def gpu_solver(datav):
  with open('./logs/.last-gpu.txt', 'r+') as fl:
    tmp = fl.read().replace('\n', '')
    os.mkdir("./logs/gpu-" + tmp)
    
    log = open("./logs/gpu-" + tmp + "/" + tmp + "-log-gpu.txt", "w")
    fl.seek(0)
    fl.write(str(int(tmp) + 1))
    fl.truncate()
    
    nameCk = "./logs/gpu-" + tmp + "/" + tmp + "-run-gpu"
  
  original = sys.stdout
  sys.stdout = Tee(sys.stdout, log)
    
  datav['X_train'] = cp.asarray(datav['X_train'])
  datav['X_val'] = cp.asarray(datav['X_val'])
  datav['y_train'] = cp.asarray(datav['y_train'])
  datav['y_val'] = cp.asarray(datav['y_val'])
  
  for i in range(1):
    start_time = timeit.default_timer()
    #lr = cp.asarray(np.random.uniform(1e-2, 1e-5))
    lr = 1e-3
    hidden_dim = 500
    # Must be odd number
    filter_size = 5
    
    # Make sure your input_dim are same order as the shape of your images.  As seen the color channels are last for me.
    # Switching them around leads to errors
    model = ConvNet_gpu(weight_scale=0.001, hidden_dim=hidden_dim, reg=0.001, filter_size=filter_size, num_filters=(23, 23, 23, 23),
                    num_classes=int(datav['y_train'].max() + 1), input_dim=(46,46, 3))

    print("lr: %e" % (lr))

    batch_size = 200
    epochs = 10
    solver = Solver_gpu(model, datav,
                    num_epochs=epochs, batch_size=batch_size,
                    update_rule="adam",
                    optim_config={"learning_rate": lr},
                    checkpoint_name=nameCk,
                    verbose=True, print_every=50,
                    num_val_samples=300, num_train_samples=400)
    
    solver.train()
    acc = solver.check_accuracy(solver.X_val, solver.y_val)
    log.write("lr: %e, acc: %f\n" % (lr, acc))
    print(">> lr: %e, acc: %f\n" % (lr, acc))
    elapsed = timeit.default_timer()
    print("<**> " + str(elapsed))
    
    log.write("Time: " + str(elapsed))
    log.write("Lr: " + str(lr))
    log.write("Batch_Size: " + str(batch_size))
    log.write("Epochs: " + str(epochs))
    log.write("Hidden_Dim: " + str(hidden_dim))
    log.write("Filter_Size: " + str(filter_size))
  
  log.close()

if __name__ == "__main__":
  inp = input("GPU (g) or CPU (c)?: ")
  
  print('Loading data...')
  
  #datav = get_veg_training_data(train="Extraction/imgs_veg/train", test="Extraction/imgs_veg/test")
  
  #Change this to whatever pickel file you want to use.
  with open('./cache/data_cache_mini.pkl','rb') as f:
    datav = pickle.load(f)
  #datav = get_CIFAR10_data()
  if inp == "g":
    gpu_solver(datav)
  else:
    datav = roll_data_channel_last(datav)
    print('Starting...')
    cpu_solver(datav)