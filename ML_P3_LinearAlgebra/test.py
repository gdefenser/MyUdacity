
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