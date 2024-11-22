''' 
This is the main file of Part 1: Julesz Ensemble
'''

from numpy.ctypeslib import ndpointer
import numpy as np
from filters import get_filters
import cv2
from scipy.signal import convolve2d
from gibbs import gibbs_sample
# from convolution import conv_cython as conv


def get_histogram(filtered_image, bins_num, max_response, min_response, img_H, img_W):
    bins = np.linspace(min_response, max_response, bins_num + 1)
    histogram, _ = np.histogram(filtered_image, bins=bins)
    histogram = histogram / (img_H * img_W)
    return histogram


class configer():
    def __init__(self, num_bin=16, save_image=True, sweep=80, T=0.63, weight=None, my_conv = False):
        self.num_bin = num_bin
        self.save_image = save_image
        self.sweep = sweep
        self.T = T
        self.weight = np.ones(num_bin) if weight is None else weight
        self.my_conv = my_conv

    def adjust_weights(self):
        for i in range(len(self.weight)//2):
            self.weight[i] = self.weight[len(self.weight) - 1 - i]= len(self.weight) - i
        # self.weight = self.weight / np.sum(self.weight)
    


def julesz(img_size=64, img_name="fur_obs.jpg", save_img=True):
    ''' 
    The main method for generating a synthesized image using the Julesz ensemble synthesis method.
    
    Parameters:
        img_size (int): The size of the image.
        img_name (str): The name of the image file.
        save_img (bool): Whether to save intermediate results (for autograder).
    
    Returns:
        np.ndarray: The synthesized image.
    '''
    
    # Initialize configuration
    config = configer(16, True, my_conv=False)
    config.adjust_weights()
    print("Current weights:", config.weight)
    
    max_intensity = 255

    # Get and process filters
    F_list = get_filters()
    F_list = [filter.astype(np.float32) for filter in F_list]

    # Initialize filter list
    filter_list = []
    
    # Image size
    img_H, img_W = img_size, img_size

    # Load and preprocess image
    img_ori = cv2.imread(f'images/{img_name}', cv2.IMREAD_GRAYSCALE)
    img_ori = cv2.resize(img_ori, (img_H, img_W)).astype(np.float32)
    img_ori = img_ori * 7 // max_intensity

    # Optionally save the original image
    if save_img:
        cv2.imwrite(f"results/{img_name.split('.')[0]}/original.jpg", (img_ori / 7 * 255))
    
    # Initialize synthesized image with random noise
    img_syn = np.random.randint(0, 8, img_ori.shape).astype(np.float32)
    
    max_error = 50  # Initial error
    threshold = 1e-3  # Convergence threshold
    round_num = 0
    
    # Initialize data structures for bounds and histograms
    bounds = []
    hists_chosen_ori = []
    hists_chosen_syn = []

    print("---- Julesz Ensemble Synthesis ----")
    
    # Synthesis loop until convergence or max iterations
    while max_error > threshold and F_list and len(filter_list) < 12:
        # Prepare to calculate the convolution responses
        conved_images_ori = []
        conved_images_syn = []

        # Convolve with filters
        for filter in F_list:
            if config.my_conv:
                conved_images_ori.append(conv(img_ori, filter))
                conved_images_syn.append(conv(img_syn, filter))
            else:
                conved_images_ori.append(convolve2d(img_ori, filter, mode='same'))
                conved_images_syn.append(convolve2d(img_syn, filter, mode='same'))

        print(f"Number of convolved images: {len(conved_images_ori)}")
        
        # Compute histograms
        hists_ori, hists_syn = [], []
        for conved_image_ori, conved_image_syn in zip(conved_images_ori, conved_images_syn):
            max_response = conved_image_ori.max()
            min_response = conved_image_ori.min()
            hists_ori.append(get_histogram(conved_image_ori, config.num_bin, max_response, min_response, img_H, img_W))
            hists_syn.append(get_histogram(conved_image_syn, config.num_bin, max_response, min_response, img_H, img_W))
        
        print(f"Number of filters: {len(F_list)}")
        print(f"Number of histograms: {len(hists_ori)}")
        
        # Calculate error
        error = cal_error(hists_ori, hists_syn)
        max_error = error.max()
        max_error_idx = error.argmax()

        # Add selected filter to filter list and remove it from F_list
        filter_list.append(F_list[max_error_idx])
        bounds.append([conved_images_ori[max_error_idx].max(), conved_images_ori[max_error_idx].min()])
        del F_list[max_error_idx]
        
        print(f"New filter selected (index {max_error_idx})")

        # Save selected filter index
        if save_img:
            with open(f"results/{img_name.split('.')[0]}/filter_list.txt", "a") as f:
                f.write(f"{max_error_idx}\n")

        # Convolve with selected filter
        chosen_filter = filter_list[-1]
        if config.my_conv:
            conved_choosen_ori = conv(img_ori, chosen_filter)
            conved_choosen_syn = conv(img_syn, chosen_filter)
        else:
            conved_choosen_ori = convolve2d(img_ori, chosen_filter, mode='same')
            conved_choosen_syn = convolve2d(img_syn, chosen_filter, mode='same')

        # Compute histograms for the selected filter
        hist_chosen_ori = get_histogram(conved_choosen_ori, config.num_bin, conved_choosen_ori.max(), conved_choosen_ori.min(), img_H, img_W)
        hist_chosen_syn = get_histogram(conved_choosen_syn, config.num_bin, conved_choosen_ori.max(), conved_choosen_ori.min(), img_H, img_W)
        hists_chosen_ori.append(hist_chosen_ori)
        hists_chosen_syn.append(hist_chosen_syn)
        
        # Perform Gibbs sampling
        img_syn, hists_chosen_syn = gibbs_sample(
            img_syn, hists_chosen_syn, hists_chosen_ori, filter_list,
            config.sweep, bounds, config.T, config.weight, config.num_bin, config.my_conv
        )

        # Save the synthesized image
        if save_img:
            synthetic = img_syn / 7 * 255
            cv2.imwrite(f"results/{img_name.split('.')[0]}/synthesized_{round_num}.jpg", synthetic)

        # Log round and error
        print(f"Round: {round_num}, Max error: {max_error}")
        if save_img:
            with open(f"results/{img_name.split('.')[0]}/error.txt", "a") as f:
                f.write(f"{max_error}\n")
        
        round_num += 1

    return img_syn  # Return synthesized image for testing
def cal_error(hist_ori:list, hist_syn:list)->np.ndarray:
    error = np.zeros(len(hist_ori))
    idx = 0
    for ori, syn in zip(hist_ori, hist_syn):
        error[idx] = np.sqrt(np.sum((ori - syn) ** 2))
        idx += 1
    return error

if __name__ == '__main__':
    image_name_list = ["grass_obs.jpg"]
    for name in image_name_list:
        julesz(img_name=name, save_img=True)


import numpy as np
cimport numpy as np

def conv(image: np.ndarray[np.float32_t, ndim=2], filt: np.ndarray[np.float32_t, ndim=2]) -> np.ndarray[np.float32_t, ndim=2]:
    """
    Perform convolution of an image with a filter (kernel).
    
    Parameters:
        image (np.ndarray): The input image to convolve, of shape (H, W).
        filt (np.ndarray): The convolution filter, of shape (Fh, Fw).
    
    Returns:
        np.ndarray: The convolved image.
    """
    # Padding image to handle borders
    padded_image = padding(image, filt)
    H, W = padded_image.shape
    Fh, Fw = filt.shape
    convolved_image = np.zeros((H - Fh + 1, W - Fw + 1), dtype=np.float32)

    # Efficient implementation using sliding window
    for i in range(Fh // 2, H - Fh // 2):
        for j in range(Fw // 2, W - Fw // 2):
            # Extract the region of interest for convolution
            region = padded_image[i - Fh // 2:i + Fh // 2 + 1, j - Fw // 2:j + Fw // 2 + 1]
            # Perform element-wise multiplication and sum
            convolved_image[i - Fh // 2, j - Fw // 2] = np.sum(region * filt)

    return convolved_image

cpdef np.ndarray padding(np.ndarray[np.float32_t, ndim=2] image, np.ndarray[np.float32_t, ndim=2] filt):
    if filt.shape[0] == 1:
        image = np.pad(image, ((0, 0), (1, 0)), 'wrap')
    elif filt.shape[0] == 2:
        image = np.pad(image, ((1, 0), (0, 0)), 'wrap')
    else:
        image = np.pad(image, ((filt.shape[0] // 2, filt.shape[0] // 2), (filt.shape[1] // 2, filt.shape[1] // 2)), 'wrap')
    return image
