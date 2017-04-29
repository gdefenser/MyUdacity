import math
def shape(M):
    return len(M[0]),len(M)

def matxRound(M, decPts=4):
    for row in range(len(M)):
        for col in range(len(M[row])):
            M[row][col] = round(M[row][col],decPts)

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
        M[r] = [elm*scale for elm in M[r]]

def addScaledRow(M, r1, r2, scale):
    if scale != 0:
        M[r1] = [elm1+elm2*scale for elm1,elm2 in zip(M[r1],M[r2])]


def gj_Solve(A, b, decPts=4, epsilon=1.0e-16):
    len_A = len(A)
    len_b = len(b)
    if len_A == len_b:
        matrix = augmentMatrix(A,b)
        for c in range(0,1):
            max = findUnderDiagonalMaximumRow(matrix,c,len(matrix))
            max_row = max[0]
            max_col = max[1]
            max_elm = max[2]
            if max_elm == 0 :
                return None
            else:
                swapRows(matrix,c,max_row)
                s = -matrix[c][c]/max_elm
                print s
                scaleRow(matrix,max_row,s)
                for row in range(len(matrix)):
                    if c != row and matrix[row][c] != 0:
                        addScaledRow(matrix,row,c,s)
        return matrix
    return None

def findUnderDiagonalMaximumRow(A,col,row_range):
    max = [0]*3
    for row in range(col,row_range):
        elm = abs(A[row][col])
        if elm > abs(max[2]):
            max[0] = row
            max[1] = col
            max[2] = elm
    #print max
    return max

A = [[0,1,1],
     [1,-1,1],
     [1,2,5]]
b = [[1],
     [2],
     [3]]

Ab = gj_Solve(A,b)
for row in range(len(Ab)):
    print Ab[row]
#print findUnderDiagonalMaximumRow(A,0,3)
