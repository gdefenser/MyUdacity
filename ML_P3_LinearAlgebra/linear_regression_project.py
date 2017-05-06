
# coding: utf-8

# # 1 矩阵运算
# 
# ## 1.1 创建一个 4*4 的单位矩阵

# In[28]:

# 这个项目设计来帮你熟悉 python list 和线性代数
# 你不能调用任何python库，包括NumPy，来完成作业

A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]

B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]

# 创建一个 4*4 单位矩阵
I = [[1,2,3,4],
     [2,3,3,5],
     [1,2,5,1],
     [3,4,5,6]]


# ## 1.2 返回矩阵的行数和列数

# In[29]:

# 返回矩阵的行数和列数
def shape(M):
    try:
        return len(M),len(M[0])
    except Exception:
        return len(M),1


# ## 1.3 每个元素四舍五入到特定小数数位

# In[30]:

# 每个元素四舍五入到特定小数数位
# 直接修改参数矩阵，无返回值
def matxRound(M, decPts=4):
    for row in range(len(M)):
        for col in range(len(M[row])):
            M[row][col] = round(M[row][col],decPts)


# ## 1.4 计算矩阵的转置

# In[31]:

# 计算矩阵的转置
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


# ## 1.5 计算矩阵乘法 AB

# In[32]:

# 计算矩阵乘法 AB，如果无法相乘则返回None
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


# ## 1.6 测试你的函数是否实现正确

# **提示：** 你可以用`from pprint import pprint`来更漂亮的打印数据，详见[用法示例](http://cn-static.udacity.com/mlnd/images/pprint.png)和[文档说明](https://docs.python.org/2/library/pprint.html#pprint.pprint)。

# In[33]:

import pprint

pp = pprint.PrettyPrinter(indent=1,width=20)

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
#测试1.2 返回矩阵的行和列
pp.pprint(shape(A))
#测试1.3 每个元素四舍五入到特定小数数位
matxRound(B,2)
#测试1.4 计算矩阵的转置
pp.pprint(transpose(B))
#测试1.5 计算矩阵乘法AB，AB无法相乘
I1 = [[1,2,3,4],
      [4,5,6,7],
      [8,9,10,11]]
I2 = [[1,4,8],
      [2,5,9],
      [3,6,10],
      [4,8,11]]
pp.pprint(matxMultiply(I1,I2))
#测试1.5 计算矩阵乘法AB，AB可以相乘
I3 = [[1,2,3,4,5],
      [4,5,6,7],
      [8,9,10,11]]
I4 = [[1,4,8],
      [2,5,9],
      [3,6,10],
      [4,8,11]]
pp.pprint(matxMultiply(I3,I4))


# # 2 Gaussign Jordan 消元法
# 
# ## 2.1 构造增广矩阵
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
# 返回 $ Ab = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n} & b_{1}\\
#     a_{21}    & a_{22} & ... & a_{2n} & b_{2}\\
#     a_{31}    & a_{22} & ... & a_{3n} & b_{3}\\
#     ...    & ... & ... & ...& ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn} & b_{n} \end{bmatrix}$

# In[34]:

# 构造增广矩阵，假设A，b行数相同
def augmentMatrix(A, b):
    if len(A) == len(b):
        for row in range(len(A)):
            if len(b[row]) == 1:
                A[row].append(b[row][0])
            else:
                return None
        return A
    return None


# ## 2.2 初等行变换
# - 交换两行
# - 把某行乘以一个非零常数
# - 把某行加上另一行的若干倍：

# In[35]:

# r1 <---> r2
# 直接修改参数矩阵，无返回值
def swapRows(M, r1, r2):
    M[r1],M[r2] = M[r2],M[r1]
    
# r1 <--- r1 * scale， scale!=0
# 直接修改参数矩阵，无返回值
def scaleRow(M, r, scale):
    if scale != 0:
        for c in range(len(M[r])):
            M[r][c] *= scale
            if M[r][c] == -0:
                M[r][c] = 0
        #M[r] = [elm*scale for elm in M[r]]

# r1 <--- r1 + r2*scale
# 直接修改参数矩阵，无返回值
def addScaledRow(M, r1, r2, scale):
    if scale != 0:
        for c in range(len(M[r1])):
            M[r1][c] = M[r1][c]+M[r2][c]*scale
            if M[r1][c] == -0:
                M[r1][c] = 0
        #M[r1] = [elm1+elm2*scale for elm1,elm2 in zip(M[r1],M[r2])]


# ## 2.3  Gaussian Jordan 消元法求解 Ax = b

# ### 提示：
# 
# 步骤1 检查A，b是否行数相同
# 
# 步骤2 构造增广矩阵Ab
# 
# 步骤3 逐列转换Ab为化简行阶梯形矩阵 [中文维基链接](https://zh.wikipedia.org/wiki/%E9%98%B6%E6%A2%AF%E5%BD%A2%E7%9F%A9%E9%98%B5#.E5.8C.96.E7.AE.80.E5.90.8E.E7.9A.84-.7Bzh-hans:.E8.A1.8C.3B_zh-hant:.E5.88.97.3B.7D-.E9.98.B6.E6.A2.AF.E5.BD.A2.E7.9F.A9.E9.98.B5)
#     
#     对于Ab的每一列（最后一列除外）
#         当前列为列c
#         寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值
#         如果绝对值最大值为0
#             那么A为奇异矩阵，返回None （请在问题2.4中证明该命题）
#         否则
#             使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c） 
#             使用第二个行变换，将列c的对角线元素缩放为1
#             多次使用第三个行变换，将列c的其他元素消为0
#             
# 步骤4 返回Ab的最后一列
# 
# ### 注：
# 我们并没有按照常规方法先把矩阵转化为行阶梯形矩阵，再转换为化简行阶梯形矩阵，而是一步到位。如果你熟悉常规方法的话，可以思考一下两者的等价性。

# In[36]:

# 实现 Gaussain Jordan 方法求解 Ax = b

""" Gaussian Jordan 方法求解 Ax = b.
    参数
        A: 方阵 
        b: 列向量
        decPts: 四舍五入位数，默认为4
        epsilon: 判读是否为0的阈值，默认 1.0e-16
        
    返回列向量 x 使得 Ax = b 
    返回None，如果 A，b 高度不同
    返回None，如果 A 为奇异矩阵
"""

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


# ## 2.4 证明下面的命题：
# 
# **如果方阵 A 可以被分为4个部分: ** 
# 
# $ A = \begin{bmatrix}
#     I    & X \\
#     Z    & Y \\
# \end{bmatrix} , \text{其中 I 为单位矩阵，Z 为全0矩阵，Y 的第一列全0}$，
# 
# **那么A为奇异矩阵。**
# 
# 提示：从多种角度都可以完成证明
# - 考虑矩阵 Y 和 矩阵 A 的秩
# - 考虑矩阵 Y 和 矩阵 A 的行列式
# - 考虑矩阵 A 的某一列是其他列的线性组合

# ### 使用行列式证明
# 
# 假设有矩阵A
# 
# $
# A = \begin{bmatrix}
#     a_{11}&a_{12}& ... &a_{1c}\\
#     a_{21}&a_{22}& ... &a_{2c}\\
#     a_{31}&a_{22}& ... &a_{3c}\\
#     ...   & ...  & ... & ...\\
#     a_{r1}&a_{r2}& ... &a_{rc}\\
# \end{bmatrix} 
# $
# 
# 其中r为行索引,c为列索引
# 
# 因为r的最大值为矩阵的行数，c的最大值为矩阵的列数
# 
# 所以当遍历元素时，可使用r和c指定遍历范围
# 
# 所以，对于矩阵中，主对角线元素，可使用$a_{11},a_{22},...,a_{rc}$标识得出
# 
# 因为可知根据行列式的性质，对于r x c阶矩阵，其行列式公式为
# 
# $$
# D(A)=a_{11}a_{22}...a_{rc} + a_{12}a_{23}...a_{r-1c}a_{r1}... + a_{1c}a_{21}...a_{rc-1}
# $$
# $$
#   - a_{1c}a_{2c-1}...a_{r1} - a_{r-11}a_{r-22}...a_{1r-1}a_{rc} - a_{11}a_{r2}...a_{2c}
# $$
# 
# 即主对角线积-副对角线积
# 
# 因为已知子矩阵Z为全0矩阵,且除主对角线外，公式中均会乘以位于主对角线以下的区域的元素，即子矩阵Z的元素
# 
# 所以根据行列式公式定义,行列式公式可化简为
# 
# $
# D（A）=a_{11}a_{22}...a_{rc}
# $
# 
# 因为子矩阵Y位于主矩阵A的对角线上,且主矩阵A是方阵
# 
# 所以子矩阵Y的首行首列元素必然位于主矩阵A的对角线上
# 
# 因为已知子矩阵Y的首列均为0元素
# 
# 所以位于主矩阵A对角线上的子矩阵Y的首行首列元素必然为0
# 
# 所以矩阵A的行列式
# 
# $
# D（A）=0
# $
# 
# 因为奇异矩阵的定义为行列式为零的矩阵
# 
# 所以可证明，矩阵A为奇异矩阵
# 

# ## 2.5 测试 gj_Solve() 实现是否正确

# In[37]:

from decimal import Decimal
from copy import deepcopy
# 构造 矩阵A，列向量b，其中 A 为奇异矩阵
A1 = [[1,1,1],
     [0,-1,0],
     [0,0,0]]
b1 = [[1],
     [2],
     [2]]
x1 = gj_Solve(A1,b1)
pp.pprint(x1)

# 构造 矩阵A，列向量b，其中 A 为非奇异矩阵
A2 = [[1,1,1],
     [0,1,0],
     [1,1,-1]]
b2 = [[1],
     [2],
     [3]]
x2 = gj_Solve(A2,b2)
pp.pprint(x2)

# 求解 x 使得 Ax = b
def calculateAx(A,x):
    Ax = []
    for row in range(len(A)):
        row_result = 0.0
        for col in range(len(A[row])):
            row_result += A[row][col]*x[col][0]
        Ax.append([row_result])
    return Ax

# 计算 Ax
Ax2 = calculateAx(A2,x2)
pp.pprint(Ax2)

# 比较 Ax 与 b
def isEqual(Ax,b):
    if len(Ax) == len(b):
        for row in range(len(Ax)):
            if float(Ax[row][0]) != float(b[row][0]):
                return False
    return True

pp.pprint(isEqual(Ax2,b2))


# # 3 线性回归: 
# 
# ## 3.1 计算损失函数相对于参数的导数 (两个3.1 选做其一)
# 
# 我们定义损失函数 E ：
# $$
# E = \sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 
# 证明：
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
# \text{其中 }
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

# ### 偏导证明过程
# 
# 因为已知
# 
# $$
# E = \sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 
# 所以当对m求偏导时，可将b,x,y设成常量，因此E可分解为
# 
# $$
# E_m = \sum_{i=1}^{n}{
#         \begin{bmatrix}
#             (y_i-b)^2-2(y_i-b)mx_i+(mx_i)^2
#         \end{bmatrix}
#     } 
# $$
# 
# 所以Em的导数求解为
# 
# $$
# E_m' = \sum_{i=1}^{n}{
#         \begin{bmatrix}
#             2y-2b-2(mx_iy)'+2b(mx_i)'+2(mx_i)
#         \end{bmatrix}
#     } = \sum_{i=1}^{n}{
#         \begin{bmatrix}
#             -2(y_i-b)x_i+2mx_i^2
#         \end{bmatrix}
#     }
# $$
# 
# 所以化简后即为
# 
# $$
# E_m' = \sum_{i=1}^{n}{
#         \begin{bmatrix}
#             -2x_i(y_i-b-mx_i)
#         \end{bmatrix}
#     }
# $$
# 
# 所以所以m的偏导为
# 
# $$
# \frac{\partial E}{\partial m} = E_m' = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $$
# 
# 同理，当对b求偏导时，可将m,x,y设成常量，所以此时E可分解为
# 
# $$
# E_b = \sum_{i=1}^{n}{
#         \begin{bmatrix}
#             (y_i-mx_i)^2-2(y_i-mx_i)b+(b)^2
#         \end{bmatrix}
#     } 
# $$
# 
# 所以Eb的导数求解为
# 
# $$
# E_b' = \sum_{i=1}^{n}{
#         \begin{bmatrix}
#             -2(y_i-mx_i)+2b
#         \end{bmatrix}
#     } = \sum_{i=1}^{n}{
#         \begin{bmatrix}
#             -2(y_i-mx_i-b)
#         \end{bmatrix}
#     } 
# $$
# 
# 所以所以b的偏导为
# 
# $$
# \frac{\partial E}{\partial b} = E_b' = \sum_{i=1}^{n}{-2(y_i-mx_i-b)} 
# $$
# 
# ### 向量证明过程
# 
# 因为已知
# 
# $$
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix}
# ,
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# $$
# 
# 所以转置矩阵$X^T$为
# 
# $$
# X^T =  \begin{bmatrix}
#     x_1 & x_2 & ... & x_n \\
#     1 & 1 & ... & 1
# \end{bmatrix}
# $$
# 
# 所以相乘矩阵$X^TX$和$X^TY$分别为
# 
# $$
# X^TX = \begin{bmatrix}
#         \sum_{i=1}^{n}{x_i^2} & \sum_{i=1}^{n}{x_i}\\
#         \sum_{i=1}^{n}{x_i} & \sum_{i=1}^{n}{1}
# \end{bmatrix}
# ,
# X^TY = \begin{bmatrix}
#         \sum_{i=1}^{n}{x_iy_i} \\
#         \sum_{i=1}^{n}{y_i} 
# \end{bmatrix}
# $$
# 
# 因为$h$为
# 
# $$
# h = \begin{bmatrix}
#         m\\
#         b
# \end{bmatrix}
# $$
# 
# 所以$X^TXh$为
# 
# $$
# X^TXh = \begin{bmatrix}
#         \sum_{i=1}^{n}{x_i^2} & \sum_{i=1}^{n}{x_i}\\
#         \sum_{i=1}^{n}{x_i} & \sum_{i=1}^{n}{1}
# \end{bmatrix}
# \begin{bmatrix}
#         m\\
#         b
# \end{bmatrix} 
# $$
# $$
# = \begin{bmatrix}
#         (\sum_{i=1}^{n}{x_i^2})m+(\sum_{i=1}^{n}{x_i})b\\
#         (\sum_{i=1}^{n}{x_i})m+(\sum_{i=1}^{n}{1})b
#   \end{bmatrix}
# $$
# 
# 所以$2X^TXh-2X^TY$为
# 
# $$
# 2\begin{bmatrix}
#         (\sum_{i=1}^{n}{x_i^2})m+(\sum_{i=1}^{n}{x_i})b\\
#         (\sum_{i=1}^{n}{x_i})m+(\sum_{i=1}^{n}{1})b
#   \end{bmatrix} 
# -
# 2\begin{bmatrix}
#         \sum_{i=1}^{n}{x_iy_i} \\
#         \sum_{i=1}^{n}{y_i} 
# \end{bmatrix}
# $$
# $$
# =
# \begin{bmatrix}
#         \sum_{i=1}^{n}{2mx_i^2+2bx_i}\\
#         \sum_{i=1}^{n}{2mx_i+2b}
#   \end{bmatrix} 
# -
# \begin{bmatrix}
#         \sum_{i=1}^{n}{2x_iy_i} \\
#         \sum_{i=1}^{n}{2y_i} 
# \end{bmatrix}
# $$
# $$
# =
# \begin{bmatrix}
#         \sum_{i=1}^{n}{2mx_i^2+2bx_i-2x_iy_i}\\
#         \sum_{i=1}^{n}{2mx_i+2b-2y_i}
# \end{bmatrix} 
# $$
# $$
# =
# \begin{bmatrix}
#         \sum_{i=1}^{n}{-2x_i(y_i-mx_i-b)}\\
#         \sum_{i=1}^{n}{-2(y_i-mx_i-b)}
# \end{bmatrix} 
# $$
# 
# 因为由上述证明可得
# 
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $$
# 
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-2(y_i-mx_i-b)}  
# $$
# 
# 所以可得偏导向量
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} 
# = 
# \begin{bmatrix}
#     \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}\\
#     \sum_{i=1}^{n}{-2(y_i-mx_i-b)}  
# \end{bmatrix} 
# $$
# 
# 
# 所以可证明
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = 2X^TXh - 2X^TY
# $$

# ## 3.1 计算损失函数相对于参数的导数（两个3.1 选做其一）
# 
# 证明：
# 
# $$
# E = Y^TY -2(Xh)^TY + (Xh)^TXh
# $$ 
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix}  = \frac{\partial E}{\partial h} = 2X^TXh - 2X^TY
# $$
# 
# $$ 
# \text{其中 }
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

# TODO 请使用 latex （请参照题目的 latex 写法学习）
# 
# TODO 证明：

# ## 3.2  线性回归
# 
# ### 求解方程 $X^TXh = X^TY $, 计算线性回归的最佳参数 h

# In[38]:

# 实现线性回归
'''
参数：(x,y) 二元组列表
返回：m，b

此处我暂时以numpy计算，对于求矩阵的逆矩阵，我有点疑问，
我能查到和学习到的内容，逆矩阵的定义的前提是可逆矩阵比为方阵，
但对此回归时，点阵并不是方阵，这该如何求？
麻烦提供下思路让我可以继续修改
'''
import numpy as np
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
    h = matrix_X_t_X_i*matrix_X_t_Y
    """
    return h
'''
我此处这样设置是否正确
'''
def getDimensionPoints(points):
    x = []
    y = []
    for point in points:
        x.append([point[0],1])
        y.append([point[1]])
    return x,y


# ## 3.3 Test your linear regression implementation

# In[39]:

# 构造线性函数
h = linearRegression([[1,2],[3,4],[5,6]])
# 构造 100 个线性函数上的点，加上适当的高斯噪音
import random
def getRandomPoints():
    points = []
    for i in range(0,100):
        points.append([random.randint(0,100),random.randint(0,100)])
    return points

points = getRandomPoints()
# 对这100个点进行线性回归，将线性回归得到的函数和原线性函数比较
h = linearRegression([[1,2],[3,4],[5,6]])
print h
#print np.mat(points)*h
print linearRegression(points)


# In[ ]:



