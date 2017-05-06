import math
from decimal import Decimal, getcontext
from copy import deepcopy
import pprint
import numpy as np

def shape(M):
    try:
        return len(M),len(M[0])
    except Exception:
        return len(M),1

def matxRound(M, decPts=4):
    for row in range(len(M)):
        for col in range(len(M[row])):
            M[row][col] = round(M[row][col],decPts)

def transpose(M):
    m_shape = shape(M)
    m_row_len = m_shape[0]
    m_col_len = m_shape[1]
    t_M = []

    for col in range(m_col_len):
        if m_row_len == 1 and (isinstance(M[col],int) or isinstance(M[col], float)):
            t_M.append([M[col]])
        else:
            t_M_row = []
            for row in range(m_row_len):
                t_M_row.append(M[row][col])
            t_M.append(t_M_row)
    return t_M

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
    if len(A) == len(b):
        for row in range(len(A)):
            if len(b[row]) == 1:
                A[row].append(b[row][0])
            else:
                return None
        return A
    return None

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
        result = []
        for row in range(len(Ab)):
            result.append([Ab[row][len(Ab[row])-1]])
        return result
    return None

def printMatrix(Ab):
    for row in range(len(Ab)):
        print Ab[row]
"""
A = [[0,1,1],
     [1,-1,1],
     [1,2,5]]
b = [[1],
     [2],
     [3]]
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
]



from sklearn import linear_model

def linearRegression(points):
    dp = getDimensionPoints(points)

    matrix_X = np.mat(dp[0])
    matrix_Y = np.mat(dp[1])
    matrix_X_t = matrix_X.T
    matrix_X_t_X = matrix_X_t*matrix_X
    matrix_X_t_Y = matrix_X_t*matrix_Y

    h = matrix_X_t_X.I*matrix_X_t_Y
    """
    matrix_X = dp[0]
    matrix_Y = dp[1]
    matrix_X_t = transpose(matrix_X)
    matrix_X_t_X = matxMultiply(matrix_X_t,matrix_X)
    matrix_X_t_X_i = inverse(matrix_X_t_X)
    matrix_X_t_Y = matxMultiply(matrix_X_t,matrix_Y)
    h = matrix_X_t_X_i
    """
    return h

def getDimensionPoints(points):
    x = []
    y = []
    for point in points:
        x.append([point[0],1])
        y.append([point[1]])
    return x,y

def inverse(matrix):
    t=transpose(matrix)
    return t

import random
def getRandomPoints():
    points = []
    for i in range(0,100):
        points.append(random.shuffle([random.randint(0,100),random.randint(0,100)]))
    return points

def predict(points):

    pass
h = linearRegression([[1,2],[3,4],[5,6]])
print h
points = getRandomPoints()
#print points
#print np.mat(points)*h
print linearRegression(points)



