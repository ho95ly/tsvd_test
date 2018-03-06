import numpy as np


def unfold(A, dim=1, t=False):
    # dim=1 frontal slides
    # dim=2 lateral slides
    # dim=3 horizontal slides
    if dim == 1:
        rb_num = A.shape[2]
        unfold_mat = A[:, :, 0]
        seq = list(range(1, rb_num))
        if t is True:
            seq = reversed(seq)
        for i in seq:
            unfold_mat = np.row_stack((unfold_mat, A[:, :, i]))

    if dim == 2:
        rb_num = A.shape[1]
        unfold_mat = A[:, 0, :]
        seq = list(range(1, rb_num))
        if t is True:
            seq = reversed(seq)
        for i in seq:
            unfold_mat = np.row_stack((unfold_mat, A[:, i, :]))

    if dim == 3:
        rb_num = A.shape[0]
        unfold_mat = A[0, :, :]
        seq = list(range(1, rb_num))
        if t is True:
            seq = reversed(seq)
        for i in seq:
            unfold_mat = np.row_stack((unfold_mat, A[i, :, :]))

    return unfold_mat


def fold(A, shape=None):
    return np.reshape(A, shape)


def transpose(A, dim=1):
    A_t = fold(unfold(A, dim, t=True), A.shape)
    return A_t


def circ(vec):
    # vec (type:list) is the transpose of a vector
    # if type(vec) is list:
    vec_len = len(vec)
    circ_mat = vec
    vec_tmp = vec.copy()
    for i in range(0, vec_len-1):
        ele = vec_tmp.pop()
        vec_tmp.insert(0, ele)
        circ_mat = np.column_stack((circ_mat, vec_tmp))
    return circ_mat


def circ_m(mat, shape):
    # shape is tensor shape,mat is unfold(tensor)
    base = 0
    cursor = shape[0]
    end = base + cursor
    # idx = np.zeros((shape[2]*shape[0], shape[1]), dtype=mat.dtype)
    idx = list(range(shape[2]))
    for i in range(0, shape[2]):
        idx[i] = mat[base:end, ::]
        base = end
        end = base + cursor

    # circ_mat = circ(idx)
    # circ_mat = unfold(circ_mat, dim=1)
    circ_mat = unfold(circ(idx))
    return circ_mat


def t_svd(M, shape):
    dim = len(shape)
    seq = list(range(2, dim))
    rou = 1

    U_ht = np.zeros((shape[0], shape[0], shape[2]), dtype=np.complex128)
    V_ht = np.zeros((shape[1], shape[1], shape[2]), dtype=np.complex128)
    tmp = min(shape[0], shape[1])
    S_ht = np.zeros((shape[0], shape[1], shape[2]), dtype=np.complex128)
    S_tmp = np.zeros((tmp, tmp, shape[2]))

    for i in seq:
            rou = rou * shape[i]

    D = np.fft.fft(M)

    for i in range(0, rou):
        u, s, v = np.linalg.svd(D[:, :, i])
        U_ht[:, :, i] = u
        S_tmp[:, :, i] = np.diag(s)
        if tmp == shape[0]:
            add_mat = np.zeros((shape[0], shape[1]-shape[0]))
            S_ht[:, :, i] = np.column_stack((S_tmp[:, :, i], add_mat))
        else:
            add_mat = np.zeros((shape[0]-shape[1], shape[1]))
            S_ht[:, :, i] = np.row_stack((S_tmp[:, :, i], add_mat))
        V_ht[:, :, i] = v

    # for i in range(2, rou):
    U = np.fft.ifft(U_ht)
    S = np.fft.ifft(S_ht)
    V = np.fft.ifft(V_ht)

    return U, S, V


def t_product(A, B):

    shape = (A.shape[0], B.shape[1], A.shape[2])
    uA = unfold(A, 1)
    cA = circ_m(uA, A.shape)
    uB = unfold(B, 1)
    result = np.matmul(np.transpose(cA), uB)
    tmp = fold(result, shape)
    return tmp

    # return fold(circ_m(unfold(A, 1), A.shape)* unfold(B, 1)), shape)


'''

#M_t3 = M_t2[:, :, :]
'''
A = np.random.rand(2, 4, 3)
B = np.random.rand(4, 5, 3)

'''

a = circ([1, 2, 3, 4])
print(a)
'''

