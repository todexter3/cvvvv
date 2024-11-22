import numpy as np
cimport numpy as np

cpdef conv_cython(np.ndarray[np.float32_t, ndim=2] image, np.ndarray[np.float32_t, ndim=2] filt):
    image = padding(image, filt)
    cdef np.ndarray[np.float32_t, ndim=2] convolved_image = np.zeros((image.shape[0] - filt.shape[0] + 1, image.shape[1] - filt.shape[1] + 1), dtype=np.float32)
    cdef int i, j, m, n

    cdef int H = image.shape[0]
    cdef int W = image.shape[1]
    cdef int x = filt.shape[0]
    cdef int y = filt.shape[1]
    cdef int start_x = 0
    cdef int start_y = 0
    cdef int end_x = 0
    cdef int end_y = 0
    cdef float temp

    if x == 1:
        for i in range(H):
            for j in range(W-1):
                convolved_image[i, j] = -image[i, j]*filt[0, 0] - image[i, j + 1]*filt[0, 1]
    elif x == 2:
        for i in range(H-1):
            for j in range(W):
                convolved_image[i, j] = -image[i, j]*filt[0, 0] - image[i + 1, j]*filt[1, 0]

    else:
        for i in range(x//2, H - x//2 ):
            for j in range(y//2, W - y//2 ):
                temp = 0.0
                start_x = i - x//2
                end_x = i + x//2
                start_y = j - y//2
                end_y = j + y//2
        
                for m in range(start_x, end_x):
                    for n in range(start_y, end_y):
                        temp += image[m, n] * filt[m - start_x, n - start_y]
                convolved_image[i - x//2, j - y//2] = temp
    return convolved_image

cpdef np.ndarray padding(np.ndarray[np.float32_t, ndim=2] image, np.ndarray[np.float32_t, ndim=2] filt):
    if filt.shape[0] == 1:
        image = np.pad(image, ((0, 0), (1, 0)), 'wrap')
    elif filt.shape[0] == 2:
        image = np.pad(image, ((1, 0), (0, 0)), 'wrap')
    else:
        image = np.pad(image, ((filt.shape[0] // 2, filt.shape[0] // 2), (filt.shape[1] // 2, filt.shape[1] // 2)), 'wrap')
    return image
