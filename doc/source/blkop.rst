The :mod:`blkop` Module
=======================

Linear operators are sometimes defined by blocks. This is often the case in
numerical optimization and the solution of partial-differential equations. An
example of operator defined by blocks is

.. math::

    K =
    \begin{bmatrix}
      A & B \\
      C & D
    \end{bmatrix}

where :math:`A`, :math:`B`, :math:`C` and :math:`D` are linear operators
(perhaps themselves defined by blocks) of appropriate shape.

The general class ``BlockLinearOperator`` may be used to represent the operator
above. If more structure is present, for example if the off-diagonal blocks are
zero, :math:`K` is a block-diagonal operator and the class
``BlockDiagonalLinearOperator`` may be used to define it.

.. automodule:: blkop

General Block Operators
-----------------------

General block operators are defined using a list of lists, each of which
defines a block row. If the block operator is specified as symmetric, each
block on the diagonal must be symmetric. For example:

.. code-block:: python

    A = LinearOperator(nargin=3, nargout=3,
                    matvec=lambda v: 2*v, symmetric=True)
    B = LinearOperator(nargin=4, nargout=3, matvec=lambda v: v[:3],
                    matvec_transp=lambda v: np.concatenate((v, np.zeros(1))))
    C = LinearOperator(nargin=3, nargout=2, matvec=lambda v: v[:2],
                    matvec_transp=lambda v: np.concatenate((v, np.zeros(1))))
    D = LinearOperator(nargin=4, nargout=2, matvec=lambda v: v[:2],
                    matvec_transp=lambda v: np.concatenate((v, np.zeros(2))))
    E = LinearOperator(nargin=4, nargout=4,
                    matvec=lambda v: -v, symmetric=True)

    # Build [A  B].
    K1 = BlockLinearOperator([[A, B]])

    # Build [A  B]
    #       [C  D].
    K2 = BlockLinearOperator([[A, B], [C, D]])

    # Build [A]
    #       [C].
    K3 = BlockLinearOperator([[A], [C]])

    # Build [A  B]
    #       [B' E].
    K4 = BlockLinearOperator([[A, B], [E]], symmetric=True)


.. autoclass:: BlockLinearOperator
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

Block Diagonal Operators
------------------------

Block diagonal operators are a special case of block operators and are defined
with a list containing the blocks on the diagonal. If the block operator is
specified as symmetric, each block must be symmetric. For example:

.. code-block:: python

    K5 = BlockDiagonalLinearOperator([A, E], symmetric=True)

.. autoclass:: BlockDiagonalLinearOperator
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

Iterating and indexing
----------------------

Block operators also support iteration and indexing. Iterating over a block
operator amounts to iterating row-wise over its blocks. Iterating over a block
diagonal operator amounts to iterating over its diagonal blocks. Indexing works
as expected. Indexing general block operators requires two indices, much as
when indexing a matrix, while indexing a block diagonal operator requires a
single indices. For example:

.. code-block:: python

    K2 = BlockLinearOperator([[A, B], [C, D]])
    K2[0,:]   # Returns the block operator defined by [[A, B]].
    K2[:,1]   # Returns the block operator defined by [[C], [D]].
    K2[1,1]   # Returns the linear operator D.

    K4 = BlockLinearOperator([[A, B], [E]], symmetric=True)
    K4[0,1]   # Returns the linear operator B.T.

    K5 = BlockDiagonalLinearOperator([A, E], symmetric=True)
    K5[0]     # Returns the linear operator A.
    K5[1]     # Returns the linear operator B.
    K5[:]     # Returns the diagonal operator defines by [A, E].
