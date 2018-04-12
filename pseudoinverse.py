import numpy as np


def judge(mat_a):
    shape = mat_a .shape
    mat_e = mat_a - np.eye(shape[0])
    e_norm = np.linalg.norm(mat_e, 2)
    return e_norm**2


def sigmoid(mat):

    # mat_sig = [1/(1+np.exp(-x)) for x in mat]
    # return np.array(mat_sig).reshape(mat.shape)
    mat_sig = list(map(lambda x: 1/(1+np.exp(-x)), mat))
    return np.array(mat_sig).reshape(mat.shape)


def hyperbolic_tangent(mat):
    mat_tanh = list(map(lambda x:(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)), mat))
    return np.array(mat_tanh).reshape(mat.shape)


def pseudoinverse(x, output, error_bound=2, step=5):

    h = x
    h_pinv = np.linalg.pinv(h)
    for i in range(step):
        t = np.matmul(h, h_pinv)
        print('(h, h_pinv)', t)
        result = judge(t)
        print('result', result)
        #if result < error_bound:
        if i > 3:
            w = np.matmul(h_pinv, output)
            out = np.matmul(h, w)
            return out, h, w
        w = h_pinv
        h = sigmoid(np.matmul(h, w))
        # h = hyperbolic_tangent(np.matmul(h, w))
        h_pinv = np.linalg.pinv(h)
    print('not satisfied.')
    return False



def filter_func(x):
    if x<10e-3:
        x=0
        return x
    return x


x = [[0, 0, 1, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 1],
     [0, 1, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 1],
     [1, 0, 0, 0, 0, 1, 0, 0, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 1],
     [0, 0, 0, 0, 0, 2, 0, 0, 1],
     [1, 0, 1, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 1, 1, 0, 0, 0, 0]]


# x = np.random.rand(5, 6)
# x = np.arange(30).reshape(5, 6)
o = x

out, h, w = pseudoinverse(x, o)
out_filt = np.array(list(map(lambda x: np.round(x), out))).reshape(out.shape)
print(out_filt)
