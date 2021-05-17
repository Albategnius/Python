def Num_Stream(strParam):

  # code goes here
  temp_num = 1
  for j, num in enumerate(strParam):
    if j == 0:
      continue
    if num == strParam[j-1]:
      temp_num += 1
    else:
      temp_num = 1
    if str(temp_num) == num:
      return True
  return False

# keep this function call here 
print NumberStream(raw_input())
