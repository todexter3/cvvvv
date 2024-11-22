from julesz import *
from filters import *
from csy_gibbs import *
from convolution import conv_cython as conv
import time
import numpy as np

# test the histogram---passed
def test_histogram():
    a = np.array([[1, 2, 2.52, 3],
                 [4, 5, 1.3, 6],
                 [7, 8, 8.7, 9]])
    his = get_histogram(a, bins_num=3, max_response=10, min_response=1, img_H=1, img_W=10)
    print(his)

# test the filter function---passed
def test_get_filter():
    F_list = get_filters()
    # print(F_list[0].shape)
    # print(F_list[0].dtype)
    return F_list

# test the padding function---passed 
def test_padding(filter):
    image = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
    image_padded = padding(image, filter)
    print("origin shape: ", image.shape)
    print("filter shape: ", filter.shape)
    print("padded shape: ", image_padded.shape)
    return image_padded

# test the convolution function---passed
def test_convolve(filter):
    from scipy.signal import convolve2d
    img_ori = cv2.resize(cv2.imread(f'images/fur_obs.jpg', cv2.IMREAD_GRAYSCALE), (64, 64))

    img_ori = img_ori * 7 // 255
    img_ori = np.random.randint(0, 10, (1000, 1000))
    img_ori = (img_ori).astype(np.float32)
    # img_ori = np.array([[1, 2, 3],
    #                     [4, 5, 6],
    #                     [7, 8, 9]], dtype=np.float32)
    

    ground_truth = convolve2d(img_ori, filter, mode='same', boundary='wrap')

    print("the origin image: ", img_ori)
    print("the ground truth is", ground_truth)


    return ground_truth


# test the cal_error function---passed
def test_error():
    syn = [np.array([1, 2, 3]), np.array([4, 5, 7]), np.array([7, 8, 9])]
    ori = [np.array([4, 5, 7]), np.array([1, 2, 3]), np.array([7, 8, 9])]
    error = cal_error(syn, ori)
    print(error)
    return error

# test the config-adjust_weight function---passed
def test_config():
    config = configer(15, True)
    config.adjust_weights()
    print(config.weight)







if __name__ == '__main__':
    filters = test_get_filter()
    test_convolve(np.array(filters[28], dtype=np.float32))
