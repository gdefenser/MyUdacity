import math
from decimal import Decimal, getcontext
from copy import deepcopy
import pprint
import numpy as np

def shape(M):
    try:
        return len(M),len(M[0])
    except Exception:
        raise TypeError('Invalid matrix')

def matxRound(M, decPts=4):
    for row in range(len(M)):
        for col in range(len(M[row])):
            M[row][col] = round(M[row][col],decPts)

def transpose(M):
    m_shape = shape(M)
    m_row_len = m_shape[0]
    m_col_len = m_shape[1]
    t_M = [[M[row][col] for row in range(m_row_len)] for col in range(m_col_len)]
    return t_M

def t(t_M,row,col):
    t_M[row][col],t_M[col][row] = t_M[col][row],t_M[row][col]

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

def augmentMatrix(A, b):
    aM = deepcopy(A)
    for row in range(len(A)):
        if len(b[row]) == 1:
            aM[row].append(b[row][0])
        else:
            return None
    return aM

def swapRows(M, r1, r2):
    M[r1],M[r2] = M[r2],M[r1]

def scaleRow(M, r, scale):
    if scale != 0:
        for c in range(len(M[r])):
            M[r][c] *= scale
        #M[r] = [elm*scale for elm in M[r]]
    else:
        raise ValueError('Scale value can not be zero')

def addScaledRow(M, r1, r2, scale):
    if scale != 0:
        for c in range(len(M[r1])):
            M[r1][c] = M[r1][c]+M[r2][c]*scale
    else:
        raise ValueError('Scale value can not be zero')
        #M[r1] = [elm1+elm2*scale for elm1,elm2 in zip(M[r1],M[r2])]

def gj_Solve(A, b, decPts=4, epsilon=1.0e-16):
    mA = deepcopy(A)
    mb = deepcopy(b)
    len_A = len(mA)
    len_b = len(mb)
    if len_A == len_b:
        matrix = augmentMatrix(mA,mb)
        floatMatrix(matrix,decPts)
        for c in range(0,len(matrix)):
            max_d = findUnderDiagonalMaximumRow(matrix,c,len(matrix))
            max_row = max_d[0]
            max_col = max_d[1]
            max_elm = max_d[2]
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

def isZero(value, eps=1.0e-16):
    return abs(Decimal(value)) < eps

def calculateAx(A,x):
    Ax = []
    for row in range(len(A)):
        row_result = 0.0
        for col in range(len(A[row])):
            row_result += A[row][col]*x[col][0]
        Ax.append([row_result])
    return Ax

def isEqual(Ax,b):
    if len(Ax) == len(b):
        for row in range(len(Ax)):
            if float(Ax[row][0]) != float(b[row][0]):
                return False
    return True

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

def getResult(Ab):
    if Ab is not None:
        result = [[Ab[row][len(Ab[row])-1]] for row in range(len(Ab))]
        return result
    return None

def printMatrix(Ab):
    for row in range(len(Ab)):
        print Ab[row]

A = [[1,1,1],
     [0,1,0],
     [1,1,-1]]
b = [[1],
     [2],
     [3]]
"""
     p1 = Plane(normal_vector=[1, 1, 1], constant_term=1)
p2 = Plane(normal_vector=[0, 1, 0], constant_term=2)
p3 = Plane(normal_vector=[1, 1, -1], constant_term=3)
"""


# construct A and b where A is singular
points = [
    [1,1]
    ,[2,4]
    ,[3,5]
    ,[4,2]
    ,[5,6]
    ,[7,8]
]



from sklearn import linear_model

def linearRegression(points):
    dp = getDimensionPoints(points)

    matrix_X = dp[0]
    matrix_Y = dp[1]
    matrix_X_t = transpose(matrix_X)
    matrix_X_t_X = matxMultiply(matrix_X_t,matrix_X)
    matrix_X_t_Y = getResult(matxMultiply(matrix_X_t,matrix_Y))
    h = gj_Solve(matrix_X_t_X,matrix_X_t_Y)
    return h



def getDimensionPoints(points):
    x = []
    y = []
    for point in points:
        x.append([point[0],1])
        y.append([0,point[1]])
    return x,y

def inverse(matrix):
    t=transpose(matrix)
    return t

import random
def getRandomX():
    x = []
    for i in range(0,100000):
        x.append(random.randint(0,100))
    return x

def predict(points):

    pass


m = 2
b = 2

def calculateYBylinearFunction(x,m,b):
    y = m*x+b
    return y

import random
def getRandomX():
    x = []
    for i in range(0,1000):
        x.append(random.randint(0,100))
    return x
X = getRandomX()

Y = []
for x in X:
    y = calculateYBylinearFunction(x,m,b)
    Y.append(y)

def randomY(Y):
    for i in range(len(Y)):
        Y[i] += random.random()
randomY(Y)

def getPoints(X,Y):
    points=[]
    for i in range(len(X)):
        points.append([X[i],Y[i]])
    return points

points = getPoints(X,Y)
h = linearRegression(points)

l_m = h[0]
l_b = h[1]

print 'Before m :'+str(m)+' b :'+str(b)
print 'After m : '+str(l_m)+' b :'+str(l_b)






