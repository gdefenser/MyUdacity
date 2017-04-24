from decimal import Decimal, getcontext
from copy import deepcopy

from vector import Vector
from plane import Plane

getcontext().prec = 30


class LinearSystem(object):

    ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG = 'All planes in the system should live in the same dimension'
    NO_SOLUTIONS_MSG = 'No solutions'
    INF_SOLUTIONS_MSG = 'Infinitely many solutions'

    def __init__(self, planes):
        try:
            d = planes[0].dimension
            for p in planes:
                assert p.dimension == d

            self.planes = planes
            self.dimension = d

        except AssertionError:
            raise Exception(self.ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG)


    def swap_rows(self, row1, row2):
        self[row1],self[row2] = self[row2],self[row1]


    def multiply_coefficient_and_row(self, coefficient, row):
        p = self.__getitem__(row)
        n = self[row].normal_vector
        k = self[row].constant_term
        for i in range(0,len(n)):
            n[i] *= coefficient
        k *= coefficient
        self.__setitem__(row,Plane(n,k))


    def add_multiple_times_row_to_row(self, coefficient, row_to_add, row_to_be_added_to):
        p_to_add = self.__getitem__(row_to_add)
        p_to_be_added = self.__getitem__(row_to_be_added_to)

        n_to_add = p_to_add.normal_vector
        n_to_be_added = p_to_be_added.normal_vector
        k_to_add = p_to_add.constant_term
        k_to_be_added = p_to_be_added.constant_term

        new_n = []

        for i in range(0, len(n_to_add)):
            n = n_to_add[i]*coefficient+n_to_be_added[i]
            new_n.append(n)

        new_k = k_to_add*coefficient+k_to_be_added

        self.__setitem__(row_to_be_added_to, Plane(new_n, new_k))


    def indices_of_first_nonzero_terms_in_each_row(self):
        num_equations = len(self)
        num_variables = self.dimension

        indices = [-1] * num_equations

        for i,p in enumerate(self.planes):
            try:
                indices[i] = p.first_nonzero_index(p.normal_vector)
            except Exception as e:
                if str(e) == Plane.NO_NONZERO_ELTS_FOUND_MSG:
                    continue
                else:
                    raise e

        return indices

    def compute_triangular_form(self):
        system = deepcopy(self)
        num_equations = len(system)
        num_variables = system.dimension
        j = 0
        for i in range(num_equations):
            while j < num_variables:
                c = system[i].normal_vector[j]
                if system.is_near_zero(c):
                    has_swap = system.swap_with_row_below_for_nonzero_coefficient_if_able(i,j)
                    if not has_swap:
                        j += 1
                        continue
                system.clear_coefficients_below(i,j)
                j += 1
                break
        return system

    def compute_rref(self):
        tf = self.compute_triangular_form()
        num_equations = len(tf)
        pivot_indices = tf.indices_of_first_nonzero_terms_in_each_row()
        for i in range(num_equations)[::-1]:
            j = pivot_indices[i]
            if j < 0:
                continue
            tf.scale_row_to_make_coefficient_equal_one(i,j)
            tf.clear_coefficients_below(i,j)
        return tf

    def swap_with_row_below_for_nonzero_coefficient_if_able(self,row,col):
        num_equations = len(self)
        for k in range(row+1,num_equations):
            coefficient = self[k].normal_vector[col]
            if not self.is_near_zero(coefficient):
                self.swap_rows(row,k)
                return True
        return False

    def clear_coefficients_below(self,row,col):
        num_equations = len(self)
        beta = self[row].normal_vector[col]
        for k in range(row+1,num_equations):
            n = self[k].normal_vector
            gamma = n[col]
            alpha = -gamma/beta
            self.add_multiple_times_row_to_row(alpha,row,k)

    def scale_row_to_make_coefficient_equal_one(self,row,col):
        n = self[row].normal_vector
        beta = 1.0/n[col]
        self.multiply_coefficient_and_row(beta,row)

    def __len__(self):
        return len(self.planes)


    def __getitem__(self, i):
        return self.planes[i]


    def __setitem__(self, i, x):
        try:
            assert x.dimension == self.dimension
            self.planes[i] = x

        except AssertionError:
            raise Exception(self.ALL_PLANES_MUST_BE_IN_SAME_DIM_MSG)


    def __str__(self):
        ret = 'Linear System:\n'
        temp = ['Equation {}: {}'.format(i+1,p) for i,p in enumerate(self.planes)]
        ret += '\n'.join(temp)
        return ret

    def round(self,d):
        return round(d,3)

    def is_near_zero(self,value):
        d = MyDecimal(value)
        return d.is_near_zero()

class MyDecimal(Decimal):
    def is_near_zero(self, eps=1e-10):
        return abs(self) < eps



