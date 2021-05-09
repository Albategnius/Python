from math import exp
import numpy as np

def Log_Regr(arr):

  # code goes here
  result = []
  x,y,a,b = arr[0],arr[1],arr[2],arr[3]
  p = 1 / (1 + np.exp(-a*x-b))
  da = x*np.exp(-a*x-b) / (1+np.exp(-a*x-b))**2
  db = np.exp(-a*x-b) / (1+np.exp(-a*x-b))**2
  Lda = -y*da/p+(1-y)*da/(1-p)
  Ldb = -y*db/p+(1-y)*db/(1-p)
  result1 = round(a + Lda, 3) 
  result2 = round(b + Ldb, 3)
  
  def merge(result1,result2):
    return str(result1) +", "+ str(result2)

  return merge(result1,result2)

def func():
  hasil = Log_Regr(raw_input())
  print(hasil)

func()
