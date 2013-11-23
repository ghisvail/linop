"""Test suite for the linop module."""

from __future__ import division
from numpy.testing import TestCase, assert_, assert_equal, assert_raises
import numpy as np
import linop as lo
from linop import ShapeError


def get_matvecs(A):
    return {'shape': A.shape,
            'matvec': lambda x: np.dot(A, x),
            'rmatvec': lambda x: np.dot(A.T.conj(), x)}


def get_dtypes():
    return (np.int64, np.float64, np.complex128)


class TestLinearOperator(TestCase):
    def setUp(self):
        self.A = np.array([[1, 2, 3],
                           [4, 5, 6]])
        self.B = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
        self.C = np.array([[1, 2],
                           [3, 4]])

    def test_init(self):
        matvecs = get_matvecs(self.A)
        A = lo.LinearOperator(nargin=matvecs['shape'][1],
                                nargout=matvecs['shape'][0],
                                matvec=matvecs['matvec'])
        assert_(hasattr(A, 'matvec'))
        assert_(hasattr(A, 'dtype'))
        assert_(hasattr(A, 'H'))
        assert_(not hasattr(A, 'rmatvec'))

        A = lo.LinearOperator(nargin=matvecs['shape'][1],
                                nargout=matvecs['shape'][0],
                                matvec=matvecs['matvec'],
                                matvec_transp=matvecs['rmatvec'])
        assert_(hasattr(A, 'rmatvec'))

        A = lo.LinearOperator(nargin=matvecs['shape'][1],
                                nargout=matvecs['shape'][0],
                                matvec=matvecs['matvec'],
                                rmatvec=matvecs['rmatvec'])
        assert_(hasattr(A, 'rmatvec'))

        for dtype in get_dtypes():
            A = lo.LinearOperator(nargin=matvecs['shape'][1],
                                    nargout=matvecs['shape'][0],
                                    matvec=matvecs['matvec'],
                                    rmatvec=matvecs['rmatvec'],
                                    dtype=dtype)
            if issubclass(A.dtype, np.complex):
                assert_(not hasattr(A, 'T'))
            else:
                assert_(hasattr(A, 'T'))
                assert_(A.T is A.H)

    def test_runtime(self):
        matvecs = get_matvecs(self.A)
        A = lo.LinearOperator(nargin=matvecs['shape'][1],
                                nargout=matvecs['shape'][0],
                                matvec=matvecs['matvec'],
                                matvec_transp=matvecs['rmatvec'])

        matvecs = get_matvecs(self.B)
        B = lo.LinearOperator(nargin=matvecs['shape'][1],
                                nargout=matvecs['shape'][0],
                                matvec=matvecs['matvec'],
                                matvec_transp=matvecs['rmatvec'])

        matvecs = get_matvecs(self.C)
        C = lo.LinearOperator(nargin=matvecs['shape'][1],
                                nargout=matvecs['shape'][0],
                                matvec=matvecs['matvec'],
                                matvec_transp=matvecs['rmatvec'])

        u = np.array([1, 1])
        v = np.array([1, 1, 1])
        assert_equal(A*v, [6, 15])
        assert_equal(A*v, A.matvec([1, 1 ,1]))
        assert_equal(A.H*u, [5, 7, 9])
        assert_equal(A.H*u, A.rmatvec(u))
        assert_equal((A*2)*v, A*(2*v))
        assert_equal((A*2)*v, (2*A)*v)
        assert_equal((A/2)*v, A*(v/2))
        assert_equal((-A)*v, A*(-v))
        assert_equal((A-A)*v, [0, 0])
        assert_equal((C**2)*u, [17, 37])
        assert_equal((C**2)*u, (C*C)*u)

        assert_(isinstance(A+A, lo.LinearOperator))
        assert_(isinstance(A-A, lo.LinearOperator))
        assert_(isinstance(-A, lo.LinearOperator))
        assert_(isinstance(2*A, lo.LinearOperator))
        assert_(isinstance(A*2, lo.LinearOperator))
        assert_(isinstance(A*0, lo.ZeroOperator))
        assert_(isinstance(A/2, lo.LinearOperator))
        assert_(isinstance(C**2, lo.LinearOperator))
        assert_(isinstance(C**0, lo.IdentityOperator))

        sum_A = lambda x: A+x
        assert_raises(ValueError, sum_A, 3)
        assert_raises(ValueError, sum_A, v)
        assert_raises(ShapeError, sum_A, B)

        sub_A = lambda x: A-x
        assert_raises(ValueError, sub_A, 3)
        assert_raises(ValueError, sub_A, v)
        assert_raises(ShapeError, sub_A, B)

        mul_A = lambda x: A*x
        assert_raises(ValueError, mul_A, u)
        assert_raises(ShapeError, mul_A, A)

        div_A = lambda x: A/x
        assert_raises(ValueError, div_A, B)
        assert_raises(ValueError, div_A, u)
        assert_raises(ZeroDivisionError, div_A, 0)

        pow_A = lambda x: A**x
        pow_C = lambda x: C**x
        assert_raises(ShapeError, pow_A, 2)
        assert_raises(ValueError, pow_C, -2)
        assert_raises(ValueError, pow_C, 2.1)


class TestIdentityOperator(TestCase):
    pass


class TestDiagonalOperator(TestCase):
    pass


class TestZeroOperator(TestCase):
    pass


class TestReducedLinearOperator(TestCase):
    pass


class TestSymmetricalReducedLinearOperator(TestCase):
    pass


class TestPysparseLinearOperator(TestCase):
    pass


class TestLinearOperatorFromArray(TestCase):
    pass
