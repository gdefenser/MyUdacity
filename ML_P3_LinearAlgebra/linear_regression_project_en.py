
# coding: utf-8

# # 1 Matrix operations
# 
# ## 1.1 Create a 4*4 identity matrix

# In[ ]:

#This project is designed to get familiar with python list and linear algebra
#You cannot use import any library yourself, especially numpy

A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]

B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]

I = [[1,2,3,4],
     [2,3,3,5],
     [1,2,5,1],
     [3,4,5,6]]


# ## 1.2 get the width and height of a matrix. 

# In[ ]:

def shape(M):
    return len(M[0]),len(M)


# ## 1.3 round all elements in M to certain decimal points

# In[ ]:

def matxRound(M, decPts=4):
    for row in range(len(M)):
        for col in range(len(M[row])):
            M[row][col] = round(M[row][col],decPts)
    print M


# ## 1.4 compute transpose of M

# In[ ]:

def transpose(M):
    t_M = []
    for col in range(len(M[0])):
        n_row = []
        for row in range(len(M)):
            if len(M[row]) != len(M[0]):
                return None
            else:
                n_row.append(M[row][col])
        t_M.append(n_row)
    return t_M


# ## 1.5 compute AB. return None if the dimensions don't match

# In[ ]:

def matxMultiply(A, B):
    multiply = []

    for row_a in range(len(A)):
        multiply_row = []
        if len(A[row_a]) != len(B):
            return None
            #exit by None when cols in A not equal to rows in B
        else:
            for col_b in range(len(A)):#rows in A equal to cols in B
                if len(B[col_b]) != len(A):
                    return None # exit by None when cols in B not equal to rows in A
                col_sum = 0
                for row_b in range(len(B)):#rows in B equal to cols in A
                    col_sum += A[row_a][row_b] * B[row_b][col_b]
                multiply_row.append(col_sum)
            multiply.append(multiply_row)
    return multiply


# ## 1.6 Test your implementation

# In[ ]:

A = [[1,2,3],
     [2,3,3],
     [1,2,5]]
B = [[1.333,2.444,3.555,5.666],
     [2,3,3,5],
     [1,2,5,1]]
I = [[1,2,3,4],
     [2,3,3,5],
     [1,2,5,1],
     [3,4,5,6]]
#test the shape function
print shape(A)
#test the round function
matxRound(B,2)
#test the transpose funtion
print transpose(B)
#test the matxMultiply function, when the dimensions don't match
I1 = [[1,2,3,4],
      [4,5,6,7],
      [8,9,10,11]]
I2 = [[1,4,8],
      [2,5,9],
      [3,6,10],
      [4,8,11]]
print matxMultiply(I1,I2)
#test the matxMultiply function, when the dimensions do match
I3 = [[1,2,3,4,5],
      [4,5,6,7],
      [8,9,10,11]]
I4 = [[1,4,8],
      [2,5,9],
      [3,6,10],
      [4,8,11]]
print matxMultiply(I3,I4)


# # 2 Gaussian Jordan Elimination
# 
# ## 2.1 Compute augmented Matrix 
# 
# $ A = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n}\\
#     a_{21}    & a_{22} & ... & a_{2n}\\
#     a_{31}    & a_{22} & ... & a_{3n}\\
#     ...    & ... & ... & ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn}\\
# \end{bmatrix} , b = \begin{bmatrix}
#     b_{1}  \\
#     b_{2}  \\
#     b_{3}  \\
#     ...    \\
#     b_{n}  \\
# \end{bmatrix}$
# 
# Return $ Ab = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n} & b_{1}\\
#     a_{21}    & a_{22} & ... & a_{2n} & b_{2}\\
#     a_{31}    & a_{22} & ... & a_{3n} & b_{3}\\
#     ...    & ... & ... & ...& ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn} & b_{n} \end{bmatrix}$

# In[ ]:

def augmentMatrix(A, b):
    if len(A) == len(b):
        for row in range(len(A)):
            if len(b[row]) == 1:
                A[row].append(b[row][0])
            else:
                return None
        return A
    return None


# ## 2.2 Basic row operations
# - exchange two rows
# - scale a row
# - add a scaled row to another

# In[ ]:

def swapRows(M, r1, r2):
    M[r1],M[r2] = M[r2],M[r1]

def scaleRow(M, r, scale):
    if scale != 0:
        for c in range(len(M[r])):
            M[r][c] *= scale
            if M[r][c] == -0:
                M[r][c] = 0
        #M[r] = [elm*scale for elm in M[r]]

def addScaledRow(M, r1, r2, scale):
    if scale != 0:
        for c in range(len(M[r1])):
            M[r1][c] = M[r1][c]+M[r2][c]*scale
            if M[r1][c] == -0:
                M[r1][c] = 0
        #M[r1] = [elm1+elm2*scale for elm1,elm2 in zip(M[r1],M[r2])]


# ## 2.3  Gauss-jordan method to solve Ax = b
# 
# ### Hint：
# 
# Step 1: Check if A and b have same number of rows
# Step 2: Construct augmented matrix Ab
# 
# Step 3: Column by column, transform Ab to reduced row echelon form [wiki link](https://en.wikipedia.org/wiki/Row_echelon_form#Reduced_row_echelon_form)
#     
#     for every column of Ab (except the last one)
#         column c is the current column
#         Find in column c, at diagnal and under diagnal (row c ~ N) the maximum absolute value
#         If the maximum absolute value is 0
#             then A is singular, return None （Prove this proposition in Question 2.4）
#         else
#             Apply row operation 1, swap the row of maximum with the row of diagnal element (row c)
#             Apply row operation 2, scale the diagonal element of column c to 1
#             Apply row operation 3 mutiple time, eliminate every other element in column c
#             
# Step 4: return the last column of Ab
# 
# ### Remark：
# We don't use the standard algorithm first transfering Ab to row echelon form and then to reduced row echelon form.  Instead, we arrives directly at reduced row echelon form. If you are familiar with the stardard way, try prove to yourself that they are equivalent. 

# In[ ]:

""" Gauss-jordan method to solve x such that Ax = b.
        A: square matrix, list of lists
        b: column vector, list of lists
        decPts: degree of rounding, default value 4
        epsilon: threshold for zero, default value 1.0e-16
        
    return x such that Ax = b, list of lists 
    return None if A and b have same height
    return None if A is (almost) singular
"""
from decimal import Decimal
from copy import deepcopy

def gj_Solve(A, b, decPts=4, epsilon=1.0e-16):
    mA = deepcopy(A)
    mb = deepcopy(b)
    len_A = len(mA)
    len_b = len(mb)
    if len_A == len_b:
        matrix = augmentMatrix(mA,mb)
        floatMatrix(matrix)
        for c in range(0,len(matrix)):
            max = findUnderDiagonalMaximumRow(matrix,c,len(matrix))
            max_row = max[0]
            max_col = max[1]
            max_elm = max[2]
            if isZero(max_elm,epsilon):
                return None
            else:
                swapRows(matrix,c,max_row)
                s = float(1)/matrix[c][c]
                scaleRow(matrix,c,s)
                for row in range(len(matrix)):
                    if c != row and not isZero(matrix[row][c],epsilon):
                        s_r = -matrix[row][c]/matrix[c][c]
                        addScaledRow(matrix,row,c,s_r)
        floatMatrix(matrix)
        return matrix
    return None

def floatMatrix(Ab,decPts=2):
    for row in range(len(Ab)):
        for col in range(len(Ab[row])):
            Ab[row][col] = round(float(Ab[row][col]),decPts)

def findUnderDiagonalMaximumRow(A,col,row_range):
    max = [0]*3
    for row in range(0,row_range):
        if row >= col:
            elm = abs(A[row][col])
            if elm > abs(max[2]):
                max[0] = row
                max[1] = col
                max[2] = float(elm)
    return max

def isZero(value, eps):
    return abs(Decimal(value)) < eps

def getResult(Ab):
    if Ab is not None:
        result = []
        for row in range(len(Ab)):
            result.append([Ab[row][len(Ab[row])-1]])
        return result
    return None


# ## 2.4 Prove the following proposition:
# 
# **If square matrix A can be divided into four parts: ** 
# 
# $ A = \begin{bmatrix}
#     I    & X \\
#     Z    & Y \\
# \end{bmatrix} $, where I is the identity matrix, Z is all zero and the first column of Y is all zero, 
# 
# **then A is singular.**
# 
# Hint: There are mutiple ways to prove this problem.  
# - consider the rank of Y and A
# - consider the determinate of Y and A 
# - consider certain column is the linear combination of other columns

# **Proves**
# 
# Use determinate to judge A is or is not singular
# 
# Assume A is:
# 
# $
# A = \begin{bmatrix}
#     a_{11}&a_{12}& ... &a_{1n}\\
#     a_{21}&a_{22}& ... &a_{2n}\\
#     a_{31}&a_{22}& ... &a_{3n}\\
#     ...   & ...  & ... & ...\\
#     a_{n1}&a_{n2}& ... &a_{nn}\\
# \end{bmatrix} 
# $
# 
# Step 1:Use the Gauss-jordan method to get diagnal matrix from A:
# 
# $
# Ag = \begin{bmatrix}
#     a_{11}&0& ... &0\\
#     0&a_{22}& ... &0\\
#     0&0& ... &0\\
#     ...   & ...  & ... & ...\\
#     0&0& ... &a_{nn}\\
# \end{bmatrix} 
# $
# 
# Step 2:Follow the definition to calculate determinate,because we expected all the elements in Ag is zero except the diagonal,so the formula of determinate would be:
# 
# $
# D(Ag)=a_{11}a_{22}...a_{nn}
# $
# 
# Step 3:if D(Ag) equal to 0 then A is **singular**,otherwise is **not singular**

# **Proof**

# In[ ]:

#singular function
def isSingular(A,b):
    d_matrix = gj_Solve(A,b)
    if d_matrix is not None:
        determinate = 1.0
        for rc in range(len(d_matrix)):
            determinate *= d_matrix[rc][rc]
        if isZero(determinate) :
            return True
        else:
            return False
    return True
# construct A and b where A is singular
A1 = [[1,1,1],
     [0,-1,0],
     [0,0,0]]
b1 = [[1],
     [2],
     [2]]
# construct A and b where A is not singular
A2 = [[1,1,1],
     [0,1,0],
     [1,1,-1]]
b2 = [[1],
     [2],
     [3]]

print isSingular(A1,b1)
print isSingular(A2,b2)


# ## 2.5 Test your gj_Solve() implementation

# In[ ]:

# construct A and b where A is singular
A1 = [[1,1,1],
     [0,-1,0],
     [0,0,0]]
b1 = [[1],
     [2],
     [2]]
x1 = gj_Solve(A1,b1)
# construct A and b where A is not singular
A2 = [[1,1,1],
     [0,1,0],
     [1,1,-1]]
b2 = [[1],
     [2],
     [3]]
x2 = gj_Solve(A2,b2)
# solve x for  Ax = b 
def calculateAx(A,x):
    Ax = []
    for row in range(len(A)):
        row_result = 0.0
        for col in range(len(A[row])):
            row_result += A[row][col]*x[col][0]
        Ax.append([row_result])
    return Ax
# compute Ax
Ax2 = calculateAx(A2,x2)
# compare Ax and b
def isEqual(Ax,b):
    if len(Ax) == len(b):
        for row in range(len(Ax)):
            if float(Ax[row][0]) != float(b[row][0]):
                return False
    return True

print isEqual(Ax2,b2)


# # 3 Linear Regression: 
# 
# ## 3.1 Compute the gradient of loss function with respect to parameters 
# ## (Choose one between two 3.1 questions)
# 
# We define loss funtion E as 
# $$
# E(m, b) = \sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 
# Proves that 
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $$
# 
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# $$
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = 2X^TXh - 2X^TY
# $$
# 
# $$ 
# \text{, where }
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# TODO Please use latex （refering to the latex in problem may help）
# 
# TODO Proof：
# 
# # Could you please give me any hints to me to solve this problem ? thanks

# ## 3.1 Compute the gradient of loss function with respect to parameters 
# ## (Choose one between two 3.1 questions)
# 
# Proves that 
# $$
# E = Y^TY -2(Xh)^TY + (Xh)^TXh
# $$
# 
# $$
# \frac{\partial E}{\partial h} = 2X^TXh - 2X^TY
# $$
# 
# $$ 
# \text{,where }
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# TODO Please use latex （refering to the latex in problem may help）
# 
# TODO Proof：
# 
# # Could you please give me any hints to me to solve this problem ? thanks

# ## 3.2  Linear Regression
# ### Solve equation $X^TXh = X^TY $ to compute the best parameter for linear regression.

# In[ ]:

#TODO implement linear regression 
'''
points: list of (x,y) tuple
return m and b
'''
def linearRegression(points):
    return 0,0

#Could you please give me any hints to me to solve this problem ? thanks


# ## 3.3 Test your linear regression implementation

# In[ ]:

#TODO Construct the linear function

#TODO Construct points with gaussian noise
import random

#TODO Compute m and b and compare with ground truth

#Could you please give me any hints to me to solve this problem ? thanks

