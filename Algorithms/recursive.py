def draw_lines(length, tick_label=''):
  line = '-' * length
  if tick_label:
    line += ' '+ tick_label
  print(line)
 
def interval(center_ln):
  if center_ln > 0 :
    interval(center_ln - 1)       #recursive
    draw_lines(center_ln)
    interval(center_ln -1)        #recursive
 
def ruler(num_inch,length):
  draw_lines(major_length, 'O')
  for j in range(1,1+num_inch):
    draw_interval(major_length - 1)
    draw_line(major_length, str(j))
