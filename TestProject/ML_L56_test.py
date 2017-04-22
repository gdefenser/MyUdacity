import vector

v1 = vector.Vector([8.462,7.893,-8.187])
v2 = vector.Vector([6.984,-5.975,4.778])

v3 = vector.Vector([-8.987,-9.838,5.031])
v4 = vector.Vector([-4.268,-1.861,-8.866])

v5 = vector.Vector([1.5,9.547,3.691])
v6 = vector.Vector([-6.007,0.124,5.772])

v7 = vector.Vector([3.039,1.879])
v8 = vector.Vector([0.825,2.036])

v9 = vector.Vector([-9.88,-3.264,-8.159])
v10 = vector.Vector([-2.155,-9.353,-9.473])

v11 = vector.Vector([3.009,-6.172,3.692,-2.51])
v12 = vector.Vector([6.404,-9.144,2.759,8.718])

print v1.cross_products(v2)
print '====='
print v3.area_parallelogram(v4)
print '====='
print v5.area_triangle(v6)
print '====='
print v7.proj_b(v8)
print '====='
print v9.proj_udv(v10)
print '====='
print v11.proj_b(v12)
print v11.proj_udv(v12)
print '====='