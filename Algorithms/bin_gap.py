def binary_gap(n):
  bin_sentence = str(bin(n))[2:]
  bin_max,bin_iter = 0,0

  for i in bin_sentence:
    if i == '1':
      if bin_max < bin_iter:
        bin_max = bin_iter
        bin_iter = 0
    else:
      bin_iter += 1
  return bin_max
