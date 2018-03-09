import kmeans as km 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import random as rand
import math as math
import sys
import os

def myFind(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def main():
  try:
      inputFile = datFile = sys.argv[1]
      k = sys.argv[2]
  except Exception as e:
      print('Oh No! => %s' %e)
      print('Usage:\npython3 ./main.py <data.mat> <k>')
      sys.exit(2)
  
  mat = spio.loadmat(inputFile, squeeze_me=True)
  fname = os.path.splitext(inputFile)
  rawdata = mat[fname[0]]
  print(fname[0])
  dataIn = rawdata[:,(0,1)]
  mu, clusters = km.kmeans(int(k),10,0.00000001, dataIn)

  colors = ["r", "b", "g", "k","c","y","m"]
  for i in range(0,max(clusters)+1):
    indices = myFind(clusters,lambda x: x == i)
    x = dataIn[indices,0]
    y = dataIn[indices,1]
    plt.scatter(x,y,c=colors[i])
    plt.scatter(mu[i,0],mu[i,1],c="yellow",marker="*",s=300)
    plt.title("K-means results")
  plt.show()

  
main()