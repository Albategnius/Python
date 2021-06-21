# Fitting y = a + bx to given n data points
import numpy as np

# Input data with size n
n = int(input("How many data points? "))

# Creating numpy array x & y to store n data points
x = np.zeros(n)
y = np.zeros(n)

# Reading data
print("Enter data:")
for i in range(n):
    x[i] = float(input("x["+str(i)+"]= "))
    y[i] = float(input("y["+str(i)+"]= "))
    
# Finding required sum for least square methods
sumX,sumX2,sumY,sumXY = 0,0,0,0
for i in range(n):
    sumX = sumX + x[i]
    sumX2 = sumX2 + x[i]*x[i]
    sumY = sumY + y[i]
    sumXY = sumXY + x[i]*y[i]

# Finding coefficients a and b
b = (n*sumXY-sumX*sumY)/(n*sumX2-sumX*sumX)
a = (sumY - b*sumX)/n

# Displaying coefficients a, b & equation
print("\nCoefficients are:")
print("a: ", a)
print("b: ", b)
print("And equation is: y = %0.4f + %0.4f x" %(a,b))
