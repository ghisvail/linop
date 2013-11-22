"""Test suite for the linop module."""

from __future__ import division
from numpy.testing import TestCase, assert_, assert_equal, assert_raises
import numpy as np
import linop as lo


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

        x1 = np.array([1, 1, 1])
        A_x1 = np.array([6, 15])
        assert_equal(A*x1, A_x1)
        assert_equal(A.matvec(x1), A_x1)
        assert_equal(A.matvec(x1.tolist()), A_x1)

        x2 = np.array([1, 1])
        AT_x2 = np.array([5, 7, 9])
        assert_equal(A.T*x2, AT_x2)

        assert_(isinstance(2*A, lo.LinearOperator))
        assert_equal((2*A)*x1, 2*A_x1)

        assert_(isinstance(A*2, lo.LinearOperator))
        assert_equal((2*A)*x1, (A*2)*x1)

        assert_(isinstance(-A, lo.LinearOperator))
        assert_equal((-A)*x1, [-6, -15])

        assert_(isinstance(A-A, lo.LinearOperator))
        assert_equal((A-A)*x1, [0, 0])

        #assert_(isinstance(A/2, lo.LinearOperator))
        #assert_equal((A/3)*x1, [2, 5])

        #assert_(isinstance(A**2, lo.LinearOperator))
        #assert_equal((A**2)*x1, [36, 225])


    def test_errors(self):
        pass


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
