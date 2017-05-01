from decimal import Decimal, getcontext

from vector import Vector

getcontext().prec = 30


class Line(object):

    NO_NONZERO_ELTS_FOUND_MSG = 'No nonzero elements found'

    def __init__(self, normal_vector=None, constant_term=None):
        self.dimension = 2

        if not normal_vector:
            all_zeros = [0]*self.dimension
            normal_vector = Vector(all_zeros)
        self.normal_vector = normal_vector

        if not constant_term:
            constant_term = 0
        self.constant_term = constant_term

        self.set_basepoint()


    def set_basepoint(self):
        try:
            n = self.normal_vector
            c = self.constant_term
            basepoint_coords = [0]*self.dimension

            initial_index = Line.first_nonzero_index(n)
            initial_coefficient = n[initial_index]

            basepoint_coords[initial_index] = self.round(c/initial_coefficient)
            self.basepoint = Vector(basepoint_coords)

        except Exception as e:
            if str(e) == Line.NO_NONZERO_ELTS_FOUND_MSG:
                self.basepoint = None
            else:
                raise e


    def __str__(self):

        num_decimal_places = 3

        def write_coefficient(coefficient, is_initial_term=False):
            coefficient = round(coefficient, num_decimal_places)
            if coefficient % 1 == 0:
                coefficient = int(coefficient)

            output = ''

            if coefficient < 0:
                output += '-'
            if coefficient > 0 and not is_initial_term:
                output += '+'

            if not is_initial_term:
                output += ' '

            if abs(coefficient) != 1:
                output += '{}'.format(abs(coefficient))

            return output

        n = self.normal_vector

        try:
            initial_index = Line.first_nonzero_index(n)
            terms = [write_coefficient(n[i], is_initial_term=(i==initial_index)) + 'x_{}'.format(i+1)
                     for i in range(self.dimension) if round(n[i], num_decimal_places) != 0]
            output = ' '.join(terms)

        except Exception as e:
            if str(e) == self.NO_NONZERO_ELTS_FOUND_MSG:
                output = '0'
            else:
                raise e

        constant = round(self.constant_term, num_decimal_places)
        if constant % 1 == 0:
            constant = int(constant)
        output += ' = {}'.format(constant)

        return output

    def is_parallel(self,l):
        return  self.round_normal_vector().is_parallel(l.round_normal_vector())

    def __eq__(self,l):
        if self.is_parallel(l):
            v_s_basepoint = self.basepoint
            v_l_basepoint = l.basepoint
            d_v = Vector(v_s_basepoint.minus(v_l_basepoint.coordinates))
            return d_v.is_orthogonal(Vector(self.normal_vector))
        return False

    def convert_to_float(self,v):
        if type(v) == str or type(v) == Decimal:
            v = float(v)
        return v

    @staticmethod
    def first_nonzero_index(iterable):
        for k, item in enumerate(iterable):
            if not MyDecimal(item).is_near_zero():
                return k
        raise Exception(Line.NO_NONZERO_ELTS_FOUND_MSG)

    def round(self,d):
        return round(d,3)

    def round_normal_vector(self):
        if self.normal_vector is not Vector:
            coordinates = []
            for i in self.normal_vector:
                coordinates.append(self.round(i))
            return Vector(coordinates)
        return self.normal_vector

    def intersection(self,l):
        intersection = None
        if self.dimension == 2:
            if self.is_parallel(l) == False:
                if self != l:
                    v1 = self.normal_vector
                    v2 = l.normal_vector
                    x = self.round((v2[1]*self.constant_term-v1[1]*l.constant_term)/(v1[0]*v2[1]-v1[1]*v2[0]))
                    y = self.round((-1*v2[0]*self.constant_term+v1[0]*l.constant_term)/(v1[0]*v2[1]-v1[1]*v2[0]))
                    intersection = [x,y]
                else:
                    print self.__str__() + " and " + l.__str__() + " are same line"
            else:
                print self.__str__()+" and "+l.__str__()+" are parallel line"

        return intersection

class MyDecimal(Decimal):
    def is_near_zero(self, eps=1e-10):
        return abs(self) < eps
