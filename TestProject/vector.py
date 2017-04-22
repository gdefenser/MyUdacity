import math

class Vector(object):
    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple(coordinates)
            self.dimension = len(coordinates)

        except ValueError:
            raise ValueError('The coordinates must be nonempty')

        except TypeError:
            raise TypeError('The coordinates must be an iterable')


    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)


    def __eq__(self, v):
        return self.coordinates == v.coordinates

    def minus(self,coordinates):
        v_coordinates = []
        for i in range(0, len(self.coordinates)):
            v_coordinates.append(self.coordinates[i] - coordinates[i])
        return v_coordinates

    def dot(self,coordinates):
        v_coordinates = []
        sum_dor = 0
        for i in range(0,len(self.coordinates)):
            sum_dor += self.coordinates[i]*coordinates[i]
        return sum_dor

    def magnitude(self):
        sum_unit = 0
        for x in self.coordinates:
            sum_unit += math.pow(x,2)
        return math.sqrt(sum_unit)

    def direction_unit(self):
        v_magnitude = self.magnitude()
        v_coordinates = []
        for i in range(0,len(self.coordinates)):
            v_coordinates.append(self.coordinates[i]/v_magnitude)
        return v_coordinates

    def inner_products(self,v):
        inner_products = 0
        for i in range(0,len(self.coordinates)):
            inner_products += self.coordinates[i]*v.coordinates[i]
        return inner_products

    def radian(self,v):
        inner_products = self.inner_products(v)
        magnitude = self.magnitude() * v.magnitude()
        return math.acos(inner_products/magnitude)

    def degrees(self,v):
        return math.degrees(self.radian(v))

    def is_zero(self):
        return self.magnitude()==0

    def is_parallel(self,v):
        if self.is_zero() == False and v.is_zero() == False:
            return self.degrees(v)==180 or self.degrees(v)==0
        else:
            #zero vector parallel to every vector
            return True


    def is_orthogonal(self,v):
        if self.is_zero() == False and v.is_zero() == False:
            return self.degrees(v)==90
        else:
            #zero vector orthogonal to every vector
            return True

    def normalization(self):
        magnitude = self.magnitude()
        v_coordinates = []
        for i in range(0,len(self.coordinates)):
            v_coordinates.append(self.coordinates[i]/magnitude)
        return v_coordinates

    def proj_b(self,b):
        b_normalization = b.normalization()
        v_dot_b_normalization = self.dot(b_normalization)
        for i in range(0,len(b_normalization)):
            b_normalization[i] *= v_dot_b_normalization
        return b_normalization


    def proj_udv(self, b):
        proj_b = self.proj_b(b)
        return self.minus(proj_b)

    def cross_products(self, v):
        if len(self.coordinates) == 3 and len(v.coordinates) == 3:
            return [
                self.coordinates[1]*v.coordinates[2]-v.coordinates[1]*self.coordinates[2],
                -1*(self.coordinates[0]*v.coordinates[2]-v.coordinates[0]*self.coordinates[2]),
                self.coordinates[0]*v.coordinates[1] - v.coordinates[0]*self.coordinates[1]
                    ]
        else:
            return [0]*3

    def area_parallelogram(self,v):
        cross_products = self.cross_products(v)
        return Vector(cross_products).magnitude()

    def area_triangle(self,v):
        area_parallelogram = self.area_parallelogram(v)
        return area_parallelogram/2.0