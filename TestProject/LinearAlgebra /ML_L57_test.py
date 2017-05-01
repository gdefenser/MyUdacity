from vector import Vector
from line import Line
from plane import Plane
from linsys import LinearSystem

"""
l1 = Line([4.046,2.836] , 1.21)
l2 = Line([10.115,7.09] , 3.025)

l3 = Line([7.204,3.182] , 8.68)
l4 = Line([8.172,4.114] , 9.883)

l5 = Line([1.182,5.562] , 6.744)
l6 = Line([1.773,8.343] , 9.525)

print l1.intersection(l2)
print l1.is_parallel(l2)
print l1 == l2
print "====="
print l3.intersection(l4)
print l3.is_parallel(l4)
print l3 == l4
print "====="
print l5.intersection(l6)
print l5.is_parallel(l6)
print l5 == l6
"""

"""
p1 = Plane([-0.412,3.806,0.728],-3.46)
p2 = Plane([1.03,-9.515,-1.82],8.65)

p3 = Plane([2.611,5.528,0.283],4.6)
p4 = Plane([7.715,8.306,5.342],3.76)

p5 = Plane([-7.926,8.625,-7.217],-7.952)
p6 = Plane([-2.642,2.875,-2.404],-2.443)


print p1.is_parallel(p2)
print p1 == p2
print "====="
print p3.is_parallel(p4)
print p3 == p4
print "====="
print p5.is_parallel(p6)
print p5 == p6




p1 = Plane(normal_vector=[1,1,1], constant_term=1)
p2 = Plane(normal_vector=[0,1,1], constant_term=2)
s = LinearSystem([p1,p2])
t = s.compute_triangular_form()
if not (t[0].equation_equal(p1) and
        t[1].equation_equal(p2)):
    print 'test case 1 failed'

p1 = Plane(normal_vector=[1,1,1], constant_term=1)
p2 = Plane(normal_vector=[1,1,1], constant_term=2)
s = LinearSystem([p1,p2])
t = s.compute_triangular_form()
if not (t[0].equation_equal(p1) and
        t[1].equation_equal(Plane(constant_term=1))
        ):
    print 'test case 2 failed'

p1 = Plane(normal_vector=[1,1,1], constant_term=1)
p2 = Plane(normal_vector=[0,1,0], constant_term=2)
p3 = Plane(normal_vector=[1,1,-1], constant_term=3)
p4 = Plane(normal_vector=[1,0,-2], constant_term=2)
s = LinearSystem([p1,p2,p3,p4])
t = s.compute_triangular_form()
if not (t[0].equation_equal(p1) and
        t[1].equation_equal(p2) and
        t[2].equation_equal(Plane(normal_vector=[0,0,-2], constant_term=2)) and
        t[3].equation_equal(Plane())):
    print 'test case 3 failed'


p1 = Plane(normal_vector=[0,1,1], constant_term=1)
p2 = Plane(normal_vector=[1,-1,1], constant_term=2)
p3 = Plane(normal_vector=[1,2,-5], constant_term=3)
s = LinearSystem([p1,p2,p3])
t = s.compute_triangular_form()
if not (t[0].equation_equal(Plane(normal_vector=[1,-1,1], constant_term=2)) and
        t[1].equation_equal(Plane(normal_vector=[0,1,1], constant_term=1)) and
        t[2].equation_equal(Plane(normal_vector=[0,0,-9], constant_term=-2))):
    print 'test case 4 failed'

"""

p1 = Plane(normal_vector=[1, 1, 1], constant_term=1)
p2 = Plane(normal_vector=[0, 1, 1], constant_term=2)
s = LinearSystem([p1, p2])
print s
r = s.compute_rref()
print r
if not (r[0] == Plane(normal_vector=[1, 0, 0], constant_term=-1) and
                r[1] == p2):
    print 'test case 1 failed'

p1 = Plane(normal_vector=[1, 1, 1], constant_term=1)
p2 = Plane(normal_vector=[1, 1, 1], constant_term=2)
s = LinearSystem([p1, p2])
r = s.compute_rref()
if not (r[0] == p1 and
                r[1] == Plane(constant_term=1)):
    print 'test case 2 failed'

p1 = Plane(normal_vector=[1, 1, 1], constant_term=1)
p2 = Plane(normal_vector=[0, 1, 0], constant_term=2)
p3 = Plane(normal_vector=[1, 1, -1], constant_term=3)
p4 = Plane(normal_vector=[1, 0, -2], constant_term=2)
s = LinearSystem([p1, p2, p3, p4])
r = s.compute_rref()
if not (r[0] == Plane(normal_vector=[1, 0, 0], constant_term=0) and
                r[1] == p2 and
                r[2] == Plane(normal_vector=[0, 0, -2], constant_term=2) and
                r[3] == Plane()):
    print 'test case 3 failed'

p1 = Plane(normal_vector=[0, 1, 1], constant_term=1)
p2 = Plane(normal_vector=[1, -1, 1], constant_term=2)
p3 = Plane(normal_vector=[1, 2, -5], constant_term=3)
s = LinearSystem([p1, p2, p3])
r = s.compute_rref()
if not (r[0] == Plane(normal_vector=[1, 0, 0], constant_term=23 / 9) and
                r[1] == Plane(normal_vector=[0, 1, 0], constant_term=7 / 9) and
                r[2] == Plane(normal_vector=[0, 0, 1], constant_term=2 / 9)):
    print 'test case 4 failed'

