''' 
This is the main file of Part 1: Julesz Ensemble
'''

import numpy as np
from filters import get_filters
import cv2
from scipy.signal import convolve2d
from gibbs import lam_gibbs_sample


def get_histogram(filtered_image, bins_num, max_response, min_response, img_H, img_W):
    bins = np.linspace(min_response, max_response, bins_num + 1)
    histogram, _ = np.histogram(filtered_image, bins=bins)
    histogram = histogram / (img_H * img_W)
    return histogram


class configer():
    def __init__(self, num_bin=16, save_image=True, sweep=90, T=0.58, weight=None, my_conv = False):
        self.num_bin = num_bin
        self.save_image = save_image
        self.sweep = sweep
        self.T = T
        self.weight = np.ones(num_bin) if weight is None else weight
        self.my_conv = my_conv





def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convolve_with_filters(img, filters, use_my_conv=False):
    conved_images = []
    for filter in filters:
        if use_my_conv:
            conved_images.append(conv(img, filter))
        else:
            conved_images.append(convolve2d(img, filter, mode='same'))
    return conved_images

def compute_histograms(conved_images, num_bin, img_H, img_W):
    histograms = []
    for conved_image in conved_images:
        max_response = conved_image.max()
        min_response = conved_image.min()
        histograms.append(get_histogram(conved_image, num_bin, max_response, min_response, img_H, img_W))
    return histograms

def save_image(img, path, factor=7, max_val=255):
    img = (img / factor * max_val).astype(np.uint8)
    cv2.imwrite(path, img)

def frame(img_size=64, img_name="stone.jpg", save_img=True):
    config = configer(16, True, my_conv=False)
    max_intensity = 255
    F_list = [filter.astype(np.float32) for filter in get_filters()]
    filter_list = F_list[:6]
    img_H = img_W = img_size
    
    # Read and preprocess the original image
    img_ori = cv2.resize(cv2.imread(f'images/{img_name}', cv2.IMREAD_GRAYSCALE), (img_H, img_W))
    img_ori = img_ori.astype(np.float32)
    img_ori = (img_ori * 7 // max_intensity)
    
    # Save original image if required
    if save_img:
        ensure_dir_exists(f"results/{img_name.split('.')[0]}")
        save_image(img_ori, f"results/{img_name.split('.')[0]}/original.jpg")
    
    # Initialize synthetic image and parameters
    img_syn = np.random.randint(0, 8, img_ori.shape).astype(np.float32)
    max_error = 50  # TODO: Set threshold
    threshold = 1e-3
    
    round_num = 0
    bounds = []
    hists_chosen_ori, hists_chosen_syn = [], []
    
    print("---- Julesz Ensemble Synthesis ----")
    while max_error > threshold and len(F_list) > 0:
        # Convolve images with filters
        conved_images_ori = convolve_with_filters(img_ori, F_list, config.my_conv)
        conved_images_syn = convolve_with_filters(img_syn, F_list, config.my_conv)
        
        # Compute histograms
        hists_ori = compute_histograms(conved_images_ori, config.num_bin, img_H, img_W)
        hists_syn = compute_histograms(conved_images_syn, config.num_bin, img_H, img_W)
        
        # Compute error and select the filter with max error
        error = cal_error(hists_ori, hists_syn)
        max_error = error.max()
        max_error_idx = error.argmax()
        
        # Add selected filter to filter list and bounds
        filter_list.append(F_list[max_error_idx])
        bounds.append([conved_images_ori[max_error_idx].max(), conved_images_ori[max_error_idx].min()])
        
        # Remove the selected filter from F_list
        del F_list[max_error_idx]
        
        # Record selected filter
        if save_img:
            with open(f"results/{img_name.split('.')[0]}/filter_list.txt", "a") as f:
                f.write(f"{max_error_idx}\n")
        
        # Convolve chosen filter
        chosen_filter = filter_list[-1]
        conved_choosen_ori = convolve_with_filters(img_ori, [chosen_filter], config.my_conv)[0]
        conved_choosen_syn = convolve_with_filters(img_syn, [chosen_filter], config.my_conv)[0]
        
        # Compute histograms for chosen filter
        hist_chosen_ori = get_histogram(conved_choosen_ori, config.num_bin, conved_choosen_ori.max(), conved_choosen_ori.min(), img_H, img_W)
        hist_chosen_syn = get_histogram(conved_choosen_syn, config.num_bin, conved_choosen_ori.max(), conved_choosen_ori.min(), img_H, img_W)
        hists_chosen_ori.append(hist_chosen_ori)
        hists_chosen_syn.append(hist_chosen_syn)
        
        # Perform Gibbs sampling
        img_syn, hists_chosen_syn = lam_gibbs_sample(img_syn, hists_chosen_syn, hists_chosen_ori, filter_list, config.sweep, bounds, config.T, config.weight, config.num_bin, config.my_conv)
        
        # Save synthesized image
        if save_img:
            save_image(img_syn, f"results/{img_name.split('.')[0]}/synthesized_{round_num}.jpg")
        
        print(f"Round: {round_num}, Max error: {max_error}")
        # Record the error
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
    frame()
