import cv2
import numpy as np
from scipy import ndimage
import math
    
def get_gaussian_filter():
    [m,n] = [(ss - 1.0) / 2.0 for ss in (shape,shape)]
    [y,x] = np.ogrid[-m:m+1,-n:n+1]
    window = np.exp( -(x*x + y*y) / (2.0*sigma*sigma) )
    window[window < np.finfo(window.dtype).eps*window.max() ] = 0
    sum_window = window.sum()
    if sum_window != 0:
        window = np.divide(window, sum_window)
    return window

def lmom(X):
    (rows, cols)  = X.shape
    if cols == 1:
        X = X.reshape(1,rows)
    n = rows
    X.sort()    
    b = np.zeros(3)    
    b0 = X.mean()    
    for r in range(1,4):        
        Num = np.prod(np.tile(np.arange(r+1,n+1), (r,1))-np.tile(np.arange(1,r+1).reshape(r,1),(1,n-r)),0)        
        Num = Num.astype(np.float)                
        Den = np.prod(np.tile(n, (1, r)) - np.arange(1,r+1), 1)        
        b[r-1] = 1.0/n * sum(Num/Den * X[0,r:])
    L = np.zeros(4)
    L[0] = b0
    L[1] = 2*b[0] - b0
    L[2] = 6*b[1] - 6*b[0] + b0
    L[3] = 20*b[2] - 30*b[1] + 12*b[0] - b0
    return L

def compute_features(im):
    im = im.astype(np.float)
    window = get_gaussian_filter()
    scalenum = 2
    feat = []
    for itr_scale in range(scalenum):
        mu = cv2.filter2D(im, cv2.CV_64F, window, borderType=cv2.BORDER_CONSTANT)
        mu_sq = mu * mu
        sigma = np.sqrt(abs(cv2.filter2D(im*im, cv2.CV_64F, window, borderType=cv2.BORDER_CONSTANT) - mu_sq))        
        structdis = (im-mu)/(sigma+1)
        structdis_col_vector = np.reshape(structdis.transpose(), (structdis.size,1))
        L = lmom(structdis.reshape(structdis.size,1))
        feat = np.append(feat,[L[1], L[3]])
        shifts = [[0,1], [1,0], [1,1], [-1,1]]
        for itr_shift in shifts:
            shifted_structdis = np.roll(structdis, itr_shift[0], axis=0)
            shifted_structdis = np.roll(shifted_structdis, itr_shift[1], axis=1)

            shifted_structdis_col_vector = np.reshape(shifted_structdis.T, (shifted_structdis.size,1))
            pair = structdis_col_vector * shifted_structdis_col_vector
            L = lmom(pair.reshape(pair.size,1))
            feat = np.append(feat, L)
        im = cv2.resize(im, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    return feat


im = ndimage.imread('example.bmp', flatten=True)
feat = compute_features(im)
print feat