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
    # vec type is a list.
    # index length
    vec_len = len(vec)

    # element is a array.
    if type(vec[0]) is np.ndarray:
        idx_map = list(range(vec_len))
        circ_mat = idx_map
        vec_tmp = idx_map.copy()
        for i in range(0, vec_len - 1):
            ele = vec_tmp.pop()
            vec_tmp.insert(0, ele)
            circ_mat = np.column_stack((circ_mat, vec_tmp))
        return circ_mat

    # element is number.
    circ_mat = vec
    vec_tmp = vec.copy()
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
    raise ValueError('this function is not coded.use circ_m() function')


def circ_m(M, shape):
    # shape is tensor shape, mat is unfold(tensor)
    # shape :(a,b,c) shape[0]== 3rd dimension
    # shape[1]== 1st dimension
    # shape[2]== 2nd dimension
    base = 0
    cursor = shape[1]
    end = base + cursor
    idx = list(range(shape[0]))

    if shape[0] == 1:
        # tensor 3rd dimension is equal to 1,return the only slice directly.
        return M[0]

    # create circular sequence index
    for i in range(0, shape[0]):
        idx[i] = M[base:end, ::]
        base = end
        end = base + cursor
    circ_mat = circ(idx)  # create circular map
    '''
     if len(circ_mat.shape) == 2:
        if circ_mat.shape[0] == 1 or circ_mat.shape[1] == 1:  # vector
            circ_vec = circ(list(circ_mat))
            return circ_vec
        else:
            # matrix
            return circ_mat

    rb_num = circ_mat.shape[0]
    unfold_mat = circ_mat[0, :, :]
    seq = list(range(1, rb_num))
    for i in seq:
        unfold_mat = np.column_stack((unfold_mat, circ_mat[i, :, :]))

    return unfold_mat
    '''

    circ_idx = list(np.reshape(circ_mat, -1))
    circ_idx_tmp = circ_idx.copy()
    circ_idx_len = len(circ_idx)
    for i in range(circ_idx_len):
        pos = circ_idx[i]
        circ_idx_tmp[i] = idx[pos]

    circ_idx_tmp_array = np.array(circ_idx_tmp)

    if shape[2] == 1:  # tensor shape (n, m, 1)
        cita_sq = squeeze(circ_idx_tmp_array)
        cita_sqf = fold(cita_sq, (shape[0], shape[1], shape[0]), 3)
        cita_sqfu = unfold(cita_sqf)
        return cita_sqfu
    else:  # tensor
        part_idx_num = range(shape[0])
        base2 = 0
        cursor2 = circ_mat.shape[0]
        end2 = base2 + cursor2
        stack_idx = list(part_idx_num)

        for i in part_idx_num:
            part_tensor = circ_idx_tmp_array[base2:end2, :, :]
            part_mat = part_tensor[0, :, :]
            for j in range(1, shape[0]):
                part_mat = np.column_stack((part_mat, part_tensor[j, :, :]))

            base2 = end2
            end2 = base2 + cursor2
            stack_idx[i] = part_mat

        merge_mat = stack_idx[0]
        for i in range(1, shape[0]):
            merge_mat = np.row_stack((merge_mat, stack_idx[i]))

        return merge_mat


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


def svd_recover(mat):
    u, s, v = np.linalg.svd(mat)
    print(u)
    print(s)
    print(v)
    ss = np.diag(s)
    # ss = np.zeros((s.shape[0], 1))
    # for i in range(s.shape[0]):
    #     ss[i][0] = s[i]
    tmp = np.matmul(u, ss)
    r_mat = np.matmul(tmp, np.transpose(v))
    bbb = np.dot(u, np.dot(ss, v))
    e_mat = np.array([x-y for x, y in zip(mat, r_mat)]).reshape(mat.shape)
    bm = np.array([x - y for x, y in zip(mat, bbb)]).reshape(mat.shape)
    # return r_mat, e_mat
    return bbb, bm


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
    # WARNING: recover formula is A = Ur*Sr*transpose(Vr)
    # but check out the api document of numpy ,it does not use this formula.
    # In numpy, the correct formula is np.dot(Ur, np.dot(Sr, Vr))
    # Note the order of multiplying is opposite and Vr is not transposed.
    # I don`t understand how it works.
    # And use np.allclose() to compare original tensor and recover tensor.


def t3d_qr(A, part_keep='r'):
    # A (l,p,n)-->(n,l,p)
    if A.shape[1] < A.shape[2]:
        raise ValueError('the shape of A is incorrect. Check the shape.')
    Q_ht = np.zeros(A.shape, dtype=np.complex128)
    R_ht = np.zeros((A.shape[0], A.shape[2], A.shape[2]), dtype=np.complex128)

    A_ht = np.fft.fft(A, n=None, axis=0)
    for i in range(0, A.shape[0]):
        Q_ht[i, :, :], R_ht[i, :, :] = np.linalg.qr(A_ht[i, :, :],mode='reduced')
    Q = np.fft.ifft(Q_ht, n=None, axis=0)
    R = np.fft.ifft(R_ht, n=None, axis=0)

    if part_keep == 'r':
        return np.real(Q), np.real(R)
    else:
        return Q, R


def inside_product(A, B):
    # This definition suits for the tensor whose shape is (m,1,n)-->(n,m,1)
    if A.shape[2] != 1 or B.shape[2] != 1:
        raise ValueError('tensor shape is not correct.')
    if A.shape != B.shape:
        raise ValueError('shapes are not equal.')
    a = t_product(transpose(A), B)
    return a


def tubal_angle(A, B):

    if A.shape[2] != 1 or B.shape[2] != 1:
        raise ValueError('tensor shape is not correct.')
    if A.shape != B.shape:
        raise ValueError('shapes are not equal.')

    A_norm = t_norm(A)
    B_norm = t_norm(B)
    if A_norm == 0 or B_norm == 0:
        raise ValueError('Ask nonzero tensor.')
    k = np.dot(np.dot(A_norm, B_norm), 2)
    s = inside_product(A, B)+inside_product(B, A)
    angle = np.true_divide(np.fabs(s), k)
    return angle


def normalize(A, tol=1e-1, part_keep='r'):
    if A.shape[2] != 1:
        raise ValueError('tensor shape is not correct.')
    if t_norm(A) == 0:
        raise ValueError('Ask nonzero tensor.')
    V = np.fft.fft(A, n=None, axis=0)
    a = np.zeros((A.shape[0], 1, 1))
    for j in range(0, A.shape[0]):
        a[j, 0, 0] = np.linalg.norm(V[j, :, :], ord='fro')
        if a[j] > tol:
            V[j, :, :] = np.true_divide(V[j, :, :], a[j, 0, 0])
        else:
            V[j, :, :] = np.random.randn(A.shape[1], 1)  # why algorithm in paper set shape[0]?
            a[j, 0, 0] = np.linalg.norm(V[j, :, :], ord='fro')
            V[j, :, :] = np.true_divide(V[j, :, :], a[j, 0, 0])
            a[j, 0, 0] = 0
    V_ifft = np.fft.ifft(V, n=None, axis=0)
    a_ifft = np.fft.ifft(a, n=None, axis=0)

    if part_keep == 'r':
        return np.real(V_ifft), np.real(a_ifft)
    else:
        return V_ifft, a_ifft


def gram_schmidt(A, tol=1e-1, part_keep='r'):
    if A.shape[1] < A.shape[2]:
        raise ValueError('the shape of A is incorrect. Check the shape.')
    Q = np.zeros(A.shape)
    R = np.zeros((A.shape[0], A.shape[2], A.shape[2]))
    Q[:, :, 0:1], R[:, 0:1, 0:1] = normalize(A[:, :, 0:1], tol, part_keep)
    for i in range(1, A.shape[2]):
        X = A[:, :, i:i+1]
        for j in range(i-1):
            R[:, j:j+1, i:i+1] = t_product(transpose(Q[:, :, j:j+1]), X)
            X = X - t_product(Q[:, :, j:j+1], R[:, j:j+1, i:i+1])
        Q[:, :, i:i+1], R[:, i:i+1, i:i+1] = normalize(X, tol, part_keep)
    return Q, R

    # raise ValueError('this function is not function yet.')


def power_iteration(A, step):  # don`t understand this function.Code according to the paper Alg4.
    # A shape (n,m,m)
    if A.shape[1] != A.shape[2]:
        raise ValueError('1st and 2nd dimension is not equal.')
    V = np.random.randn(A.shape[0], A.shape[1], 1)
    v, a = normalize(V)
    V = t_product(v, a)
    for i in range(step):

        V = t_product(A, V)
        if t_norm(V) == 0:
            print('V norm is equal zero.')
            return d
        v, a = normalize(V)
        V = t_product(v, a)
        d = t_product(transpose(V), t_product(A, V))
    return d, V


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
    v, a = normalize(X, 1e-1, 'r')
    print(v)
    print(v.shape)
    print(a)
    print(a.shape)
    p = t_product(v, a)
    print(p)
    print(p.shape)
    print(X)
    
       X = np.arange(12).reshape(3, 4, 1)
    v, a = normalize(X, 1e-1, 'r')
    print(v)
    print(v.shape)
    print(a)
    print(a.shape)
    p = t_product(v, a)
    print(p)
    print(p.shape)
    print(X)

'''

if __name__ == "__main__":
    X = np.arange(24).reshape(3, 4, 2)
    # X = np.random.rand(2, 3, 3)
    a, b = gram_schmidt(X, 1e-1, 'r')
    print(a)
    print(a.shape)
    print(b)
    print(b.shape)
    aa = t_product(a,b)
    print(aa)
    print(aa.shape)

'''

 
'''
