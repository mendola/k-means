import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import random as rand
import math as math
def kmeans(k,r,thresh, data):
  n = data.shape[0]
  d = data.shape[1]
  storedMu = np.zeros((k,d,r))
  iterCount = 0
  errsSq = np.zeros((r,1))

  for iter in range(0,r):
    # Initalize mu. Here, using first c points from data
      count = 0
      iterCount += 1
      mu = np.zeros((k,d))
      newMu = np.zeros((k,d))
      dist = np.zeros((n,k))
      currClass = [0] * n

      for i in range(0,k):
        newMu[i,:] = data[math.floor((n-1)*rand.random())+1,:]

      # Perform iterative algorithm
      while True:
          count += 1

          mu[:] = newMu[:]
          # Compute distance of each point from each class mean
          for i in range(0,k):
            dist[:,i] = np.sqrt( np.sum( (mu[i,:] - data[:,:])**2, 1))
          # Classify each point using nearest mean
          classCount = [0] * k
          for i in range(0,n):
              cluster = np.argmin(dist[i,:])
              currClass[i] = cluster
              classCount[cluster] += 1
          # Recompute means using nearest mean
          newMu[:] = 0
          for i in range(0,n):
              cluster = currClass[i]
              newMu[cluster, :] = newMu[cluster, :] + data[i,:]
          for i in range(0,k):
              if classCount[i] != 0:
                newMu[i,:] = newMu[i,:] / classCount[i]

          # Compute total change in mu 
          totChange = np.sum(np.abs(newMu - mu))
          if(totChange < thresh):
              break

      mu[:] = newMu[:]
      storedMu[:,:,iter] = mu[:]
      # Compute squared errors
      errsSq[iter] = 0
      for i in range(0,n):
          cluster = currClass[i]
          errsSq[iter] = errsSq[iter] + np.sum( ( mu[cluster,:] - data[i,:] )**2)

  best = np.argmin(errsSq)
  bestMu = storedMu[:,:,best]

  for i in range(0,n):
      cluster = np.argmin(dist[i,:])
      currClass[i] = cluster
      classCount[cluster] += 1
  print(currClass)
  return bestMu,currClass
