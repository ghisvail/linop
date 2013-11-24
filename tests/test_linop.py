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
    return ((np.int64, np.int64),
            (np.uint64, np.uint64),
            (np.float64, np.float64),
            (np.complex128, np.complex128),
            (np.float32, np.float32),
            (np.uint64, np.int64),
            (np.int64, np.uint64),
            (np.float32, np.float64),
            (np.float64, np.float32),
            (np.float64, np.float32),
            (np.float64, np.complex128),
            (np.complex128, np.float64))


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

        A = lo.LinearOperator(nargin=matvecs['shape'][1],
                              nargout=matvecs['shape'][0],
                              matvec=matvecs['matvec'],
                              rmatvec=matvecs['rmatvec'],
                              dtype=np.int64)
        assert_(hasattr(A, 'T'))
        assert_(A.T is A.H)

        A = lo.LinearOperator(nargin=matvecs['shape'][1],
                              nargout=matvecs['shape'][0],
                              matvec=matvecs['matvec'],
                              rmatvec=matvecs['rmatvec'],
                              dtype=np.float64)
        assert_(hasattr(A, 'T'))
        assert_(A.T is A.H)

        A = lo.LinearOperator(nargin=matvecs['shape'][1],
                              nargout=matvecs['shape'][0],
                              matvec=matvecs['matvec'],
                              rmatvec=matvecs['rmatvec'],
                              dtype=np.complex128)
        assert_(not hasattr(A, 'T'))

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

    def test_dtypes(self):
        for dtypes in get_dtypes():
            dtype_op, dtype_in = dtypes
            dtype_out = np.result_type(dtype_op, dtype_in)

            matvecs = get_matvecs(self.A)
            A = lo.LinearOperator(nargin=matvecs['shape'][1],
                                  nargout=matvecs['shape'][0],
                                  matvec=matvecs['matvec'],
                                  matvec_transp=matvecs['rmatvec'],
                                  dtype=dtype_op)
            x = np.array([1, 1, 1]).astype(dtype_in)
            assert_((A*x).dtype == dtype_out)


class TestIdentityOperator(TestCase):
    def test_runtime(self):
        A = lo.IdentityOperator(3)
        x = np.array([1, 1, 1])
        assert_equal(A*x, x)
        assert_(A.H is A)

    def test_dtypes(self):
        for dtypes in get_dtypes():
            dtype_op, dtype_in = dtypes
            dtype_out = np.result_type(dtype_op, dtype_in)
            A = lo.IdentityOperator(3, dtype=dtype_op)
            x = np.array([1, 1, 1]).astype(dtype_in)
            assert_((A*x).dtype == dtype_out)


class TestDiagonalOperator(TestCase):
    def test_runtime(self):
        A = lo.DiagonalOperator([1, 2, 3])
        x = np.array([1, 1, 1])
        assert_equal(A*x, [1, 2, 3])
        assert_equal(A.H*x, [1, 2, 3])
        assert_(A.H is A)

    def test_dtypes(self):
        for dtypes in get_dtypes():
            dtype_op, dtype_in = dtypes
            dtype_out = np.result_type(dtype_op, dtype_in)
            diag = np.array([1, 2, 3]).astype(dtype_op)
            A = lo.DiagonalOperator(diag)
            x = np.array([1, 1, 1]).astype(dtype_in)
            assert_((A*x).dtype == dtype_out)


class TestZeroOperator(TestCase):
    def test_runtime(self):
        A = lo.ZeroOperator(2, 3)
        x = np.array([1, 1])
        assert_equal(A*x, [0, 0, 0])
        x = np.array([1, 1, 1])
        assert_equal(A.H*x, [0, 0])

    def test_dtypes(self):
        for dtypes in get_dtypes():
            dtype_op, dtype_in = dtypes
            dtype_out = np.result_type(dtype_op, dtype_in)
            A = lo.ZeroOperator(3, 3, dtype=dtype_op)
            x = np.array([1, 1, 1]).astype(dtype_in)
            assert_((A*x).dtype == dtype_out)


class TestReducedLinearOperator(TestCase):
    pass


class TestSymmetricalReducedLinearOperator(TestCase):
    pass


class TestPysparseLinearOperator(TestCase):
    pass


class TestLinearOperatorFromArray(TestCase):
    pass
