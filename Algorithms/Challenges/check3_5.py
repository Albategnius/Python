# n = input_number 
result = []
for i in range(1,n+1):
  add =''
  if i%3 ==0:
    add += 'fizz'
  if i%5 == 0:
    add += 'buzz'
  
  if add == '':
    result.append(i)
  else:
    result.append(add)
result
