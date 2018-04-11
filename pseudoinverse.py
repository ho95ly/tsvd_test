import numpy as np


def judge(mat_a):
    shape = mat_a .shape
    mat_e = mat_a - np.zeros(shape)
    e_norm = np.linalg.norm(mat_e, 'fro')
    return e_norm


def sigmoid(mat):

    # mat_sig = [1/(1+np.exp(-x)) for x in mat]
    # return np.array(mat_sig).reshape(mat.shape)
    mat_sig = list(map(lambda x: 1/(1+np.exp(-x)), mat))
    return np.array(mat_sig).reshape(mat.shape)


def pseudoinverse(x, output, error_bound=10e-3, step=10):

    h = x
    h_pinv = np.linalg.pinv(h)
    for i in range(step):
        t = np.matmul(h, h_pinv)
        print('(h, h_pinv)', t)
        result = judge(t)
        print('result', result)
        if result < error_bound:
            w = np.matmul(h_pinv, output)
            return h, w
        w = h_pinv
        h = sigmoid(np.matmul(h, w))
        h_pinv = np.linalg.pinv(h)
    print('not satisfied.')
    return False


# x = np.random.rand(3, 4)
x = np.arange(12).reshape(3, 4)
o = x

h, w = pseudoinverse(x, o)
