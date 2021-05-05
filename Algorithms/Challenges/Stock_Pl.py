def StockPickr(arr):
  # code goes here
  temp = -1
  for i in xrange(0,len(arr)):
    for j in xrange(i+1, len(arr)):
      if temp < arr[j] - arr[i]:
        temp = arr[j] - arr[i]
  return temp
