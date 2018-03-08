import numpy as np


# shape :(a,b,c) shape[0]== 3rd dimension
# shape[1]== 1st dimension
# shape[2]== 2nd dimension

def t_norm(A):
    uA = unfold(A)
    uA_norm = np.linalg.norm(uA, 'fro')
    return uA_norm


def identity_tensor(shape, data_type=np.int32):
    # I shape must be (m,m,n),and whose first frontal slice is the m×m identity matrix,
    # and whose other frontal slices are all zeros.
    Z = np.zeros(shape, data_type)
    Z[0:, :, ] = np.eye(shape[1])
    I = Z
    return I


def unfold(A, dim=1):
    # shape :(a,b,c) shape[0]== 3rd dimension
    # shape[1]== 1st dimension
    # shape[2]== 2nd dimension
    # [BASE ON] dim to slice
    # dim=1 frontal slices
    # dim=2 lateral slices
    # dim=3 horizontal slices

    if dim == 1:
        rb_num = A.shape[0]
        unfold_mat = A[0, :, :]
        seq = list(range(1, rb_num))
        for i in seq:
            unfold_mat = np.row_stack((unfold_mat, A[i, :, :]))

    if dim == 2:
        rb_num = A.shape[2]
        unfold_mat = np.transpose(A[:, :, 0])
        seq = list(range(1, rb_num))
        for i in seq:
            unfold_mat = np.row_stack((unfold_mat, np.transpose(A[:, :, i])))

    if dim == 3:
        rb_num = A.shape[1]
        unfold_mat = A[:, 0, :]
        seq = list(range(1, rb_num))
        for i in seq:
            unfold_mat = np.row_stack((unfold_mat, A[:, i, :]))

    return unfold_mat


def fold(A, shape=None, dim=1):
    if dim == 1:
        return np.reshape(A, shape)
    if dim == 2:
        A_swp02 = np.swapaxes(np.reshape(A, (shape[2], shape[1], shape[0])), 0, 2)
        return A_swp02
    if dim == 3:
        A_swp01 = np.swapaxes(np.reshape(A, (shape[1], shape[0], shape[2])), 0, 1)
        return A_swp01
    return False


def transpose(A):
    # If A is l×m×n, then At is the m×l×n tensor obtained by
    # transposing each of the frontal slices
    # and then reversing the order of transposed frontal slices 2 through n.
    rb_num = A.shape[0]
    unfold_mat = np.transpose(A[0, :, :])
    seq = reversed(list(range(1, rb_num)))
    for i in seq:
        unfold_mat = np.row_stack((unfold_mat, np.transpose(A[i, :, :])))
    shape_t = (A.shape[0], A.shape[2], A.shape[1])
    # return np.reshape(unfold_mat, shape_t)
    return fold(unfold_mat, shape_t, dim=1)


def circ(vec):
    # vec (type:list) is the transpose of a vector
    # if type(vec) is list:
    vec_len = len(vec)
    circ_mat = vec
    vec_tmp = vec.copy()

    if vec_len == 1 and type(vec[0]) is np.ndarray:
        return circ_mat[0]

    for i in range(0, vec_len-1):
        ele = vec_tmp.pop()
        vec_tmp.insert(0, ele)
        circ_mat = np.column_stack((circ_mat, vec_tmp))

    return circ_mat


def circ_v(vec, shape):
    # shape is tensor shape, mat is unfold(tensor)
    # shape :(a,b,c) shape[0]== 3rd dimension
    # shape[1]== 1st dimension
    # shape[2]== 2nd dimension
    pass


def circ_m(mat, shape):
    # shape is tensor shape, mat is unfold(tensor)
    # shape :(a,b,c) shape[0]== 3rd dimension
    # shape[1]== 1st dimension
    # shape[2]== 2nd dimension
    base = 0
    cursor = shape[1]
    end = base + cursor
    idx = list(range(shape[0]))

    for i in range(0, shape[0]):
        idx[i] = mat[base:end, ::]
        base = end
        end = base + cursor
    circ_mat = circ(idx)

    rb_num = circ_mat.shape[0]
    # if len(circ_mat.shape) == 2:
    #    circ_vec = circ(list(circ_mat))
    #    return circ_vec

    unfold_mat = circ_mat[0, :, :]
    seq = list(range(1, rb_num))
    for i in seq:
        unfold_mat = np.column_stack((unfold_mat, circ_mat[i, :, :]))

    return unfold_mat


def bcirc(A):

    uA = unfold(A, 1)
    bcA = circ_m(uA, A.shape)

    return bcA


def squeeze(A):
    # shape :(a,b,c)
    # shape[0]== 3rd dimension
    # shape[1]== 1st dimension
    # shape[2]== 2nd dimension
    # the shape of tensor A is (n,m,1)
    # A is squeezed to a (m,n) matrix!
    if A.shape[2] != 1:
        raise TypeError('shape error,the 2nd dimension is not equal to 1')
    # set the parameter order as 'F' in the reshape step for the correct result.
    A_seq = np.reshape(unfold(A, dim=1), (A.shape[1], A.shape[0]), 'F')

    return A_seq


def twist(mat):
    # matrix shape is (m,n)
    # matrix is twisted to a (m,1,n) tensor.
    # set the parameter order as 'F' in the reshape step,because squeeze function use this
    # parameter to reshape.
    mat_r = np.reshape(mat, (mat.shape[0] * mat.shape[1], 1), 'F')
    # tensor_shape = (mat.shape[1], mat.shape[0], 1)
    mat_rf = fold(mat_r, (mat.shape[1], mat.shape[0], 1), 1)

    return mat_rf


def t_product(A, B):
    # formula: A*B = fold(matmul(bcirc(A),unfold(B)))
    # shape :(a,b,c)
    # shape[0]== 3rd dimension
    # shape[1]== 1st dimension
    # shape[2]== 2nd dimension
    # so that we should write as below:
    # A(l,p,n)-->shape is (n,l,p)
    # B(p,m,n)-->shape is (n,p,m)
    # result tensor (l,m,n)-->shape is (n,l,m)
    shape = (A.shape[0], A.shape[1], B.shape[2])
    # uA = unfold(A, 1)
    # cA = circ_m(uA, A.shape)
    # cAt = np.transpose(cA)
    # uB = unfold(B, 1)
    # result = np.matmul(cAt, uB)
    # tmp = fold(result, shape)
    # return fold(circ_m(unfold(A, 1), A.shape)* unfold(B, 1)), shape)
    bA = bcirc(A)
    uB = unfold(B)
    tmp_mat = np.matmul(bA, uB)

    return fold(tmp_mat, shape)


def test_fft(Q):
    max_dim = max(list(Q.shape))
    shape_len = len(Q.shape)
    J = [j for j in range(1, max_dim+1)]
    I = [i for i in range(shape_len)]
    print(Q)
    for i in I:
        for j in J:
            print('n =', j, 'axis =', i)
            Qn = np.fft.fft(Q, n=None, axis=i)
            Q_fft = np.fft.fft(Q, n=j, axis=i)
            Qn_ifft = np.fft.ifft(Qn, n=None, axis=i)
            Q_fft_ifft = np.fft.ifft(Q_fft, n=j, axis=i)
            print('fft---------------')
            print(Q_fft.shape)
            print(Q_fft)
            print(Qn.shape)
            print(Qn)
            print('ifft++++++++++++++')
            print(Qn_ifft.shape)
            print(Qn_ifft)
            print(Q_fft_ifft.shape)
            print(Q_fft_ifft)
            print('==================')
    return True


def t3d_svd(A, part_keep=None):  # only for the three dimensions tensor
    '''
    dim = len(shape)
    seq = list(range(2, dim))
    rou = 1

    U_ht = np.zeros((shape[2], shape[1], shape[0]), dtype=np.complex128)
    V_ht = np.zeros((shape[2], shape[2], shape[0]), dtype=np.complex128)
    tmp = min(shape[1], shape[2])
    S_ht = np.zeros((shape[1], shape[2], shape[0]), dtype=np.complex128)
    S_tmp = np.zeros((tmp, tmp, shape[0]))

    for i in seq:
            rou = rou * shape[i]

    D = np.fft.fft(A)

    for i in range(0, rou):
        u, s, v = np.linalg.svd(D[i, :, :])
        U_ht[i, :, :] = u
        S_tmp[i, :, :] = np.diag(s)
        if tmp == shape[1]:
            add_mat = np.zeros((shape[1], shape[2]-shape[1]))
            S_ht[i, :, :] = np.column_stack((S_tmp[i, :, :], add_mat))
        else:
            add_mat = np.zeros((shape[1]-shape[2], shape[2]))
            S_ht[i, :, :] = np.row_stack((S_tmp[i, :, :], add_mat))
        V_ht[i, :, :] = v

    # for i in range(2, rou):
    U = np.fft.ifft(U_ht)
    S = np.fft.ifft(S_ht)
    V = np.fft.ifft(V_ht)

    return U, S, V
    '''
    # D=fft(A,[],3)
    # Axis over which to compute the FFT. If not given, the last axis is used.
    # tensor shape :(a,b,c)
    # shape[0]== 3rd dimension
    # shape[1]== 1st dimension
    # shape[2]== 2nd dimension
    # so that we should write as below:
    # A(l,p,n)-->shape is (n,l,p)
    # Axis over which to compute the FFT. If not given, the last axis is used.
    # The algorithm use 3rd dimension ,so axis is set as 0
    U = np.zeros((A.shape[0], A.shape[1], A.shape[1]), dtype=np.complex128)
    V = np.zeros((A.shape[0], A.shape[2], A.shape[2]), dtype=np.complex128)
    min_dim = min(A.shape[1], A.shape[2])
    S = np.zeros(A.shape, dtype=np.complex128)
    S_part = np.zeros((A.shape[0], min_dim, min_dim), dtype=np.complex128)

    D = np.fft.fft(A, n=None, axis=0)

    for i in range(A.shape[0]):
        u, s, v = np.linalg.svd(D[i, :, :])  # SVD algorithm
        # u shape:(A.shape[1],A.shape[1])
        # s shape:(min(A.shape[1],A.shape[2]),)
        # v shape:(A.shape[2],A.shape[2])
        U[i, :, :] = u
        S_part[i, :, :] = np.diag(s)
        if A.shape[1] == A.shape[2]:
            S[i, :, :] = S_part[i, :, :]
        elif min_dim == A.shape[1]:
            add_mat = np.zeros((A.shape[1], A.shape[2] - A.shape[1]))
            S[i, :, :] = np.column_stack((S_part[i, :, :], add_mat))
        else:
            add_mat = np.zeros((A.shape[1] - A.shape[2], A.shape[2]))
            S[i, :, :] = np.row_stack((S_part[i, :, :], add_mat))
        V[i, :, :] = v

    Ur = np.fft.ifft(U, n=None, axis=0)
    Sr = np.fft.ifft(S, n=None, axis=0)
    Vr = np.fft.ifft(V, n=None, axis=0)

    if part_keep == 'r':  # only keep real part
        return np.real(Ur), np.real(Sr), np.real(Vr)
    else:  # real part + imaginary part
        return Ur, Sr, Vr


'''
A = [[[1, 0],
      [0, 2],
      [-1, 3]],
     [[-2, 1],
      [-2, 7],
      [0, -1]]]

B = [[[3], [-1]],
     [[-2], [-3]]]

a = np.array(A)
b = np.array(B)

X = np.arange(12).reshape(3, 4, 1)
c = np.arange(3).reshape(3, 1, 1)
r = squeeze(t_product(X, c))
rr = np.matmul(squeeze(X), bcirc(transpose(c)))
print(r)
print(rr)

'''
Q = np.reshape(np.arange(24), (2, 3, 4))
a, b, c = t3d_svd(Q, 'r')
print(a.shape)
print(a)
print(b.shape)
print(b)
print(c.shape)
print(c)

print(Q)
qqq = t_product(t_product(a, b), transpose(c))
print(qqq.shape)
print(qqq.astype(np.int32))

'''

'''