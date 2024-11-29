import numpy as np

def inv_interp(input_y, x, y):
    '''
    Finds the inverse of np.interp given data pair (x,y) and target y value input_y
    '''

    #     slopes = np.zeros((x.shape[0] - 1))
    yi_target = 0
    xi_target = 0

    ndim = x.shape[0] - 1

    for xi, xi1, yi, yi1, k in zip(x, x[1:], y, y[1:], range(ndim)):
        #         slopes[k] = (yi1 - yi) / (xi1 - xi)
        #         slope = slopes[k]
        if input_y > yi and input_y < yi1:
            slope = (yi1 - yi) / (xi1 - xi)
            #             yi_target = yi
            #             xi_target = xi
            return (1 / slope) * (input_y - yi) + xi

def test_diff(x,y, iters = 1000):
    results = np.zeros(iters)
    for i in range(iters):
        a = np.random.rand()*2*np.pi
        b = np.random.rand()*2*np.pi



        fa = np.interp(a,x,y)
        fb = np.interp(b,x,y)

        inside_finv = (fa+fb) % 2*np.pi


        if inv_interp(inside_finv , x,y) == None:
            print(inside_finv )

        results[i] = np.abs(inv_interp(inside_finv, x,y) - (a+b))
    return results