''' 
This is file for part 1 
It defines the Gibbs sampler and we use cython for acceleration
'''
from tqdm import tqdm
import numpy as np
import random
# from convolution import conv_cython as conv
from matplotlib import pyplot as plt
from scipy.signal import convolve2d


def get_histogram(filtered_image, bins_num, max_response, min_response, img_H, img_W):
    bins = np.linspace(min_response, max_response, bins_num + 1)
    histogram, _ = np.histogram(filtered_image, bins=bins)
    histogram = histogram / (img_H * img_W)
    return histogram



def gibbs_sample(img_syn, hists_syn,
                 hists_ori,
                 filter_list, sweep, bounds,
                 T, weight, num_bins, my_conv):
    '''
    The gibbs sampler for synthesizing a texture image using annealing scheme
    Parameters:
        1. img_syn: the synthesized image, numpy array, shape: [H,W]
        2. hists_syn: the histograms of the synthesized image, numpy array, shape: [num_chosen_filters,num_bins]
        3. img_ori: the original image, numpy array, shape: [H,W]
        4. hists_ori: the histograms of the original image, numpy arrays, shape: [num_chosen_filters,num_bins]
        5. filter_list: the list of selected filters
        6. sweep: the number of sweeps
        7. bounds: the bounds of the responses of img_ori, a array of numpy arrays in shape [num_chosen_filters,2], bounds[x][0] max response, bounds[x][1] min response
        8. T: the initial temperature
        9. weight: the weight of the error, a numpy array in the shape of [num_bins]
        10. num_bins: the number of bins of histogram, a scalar
    Return:
        img_syn: the synthesized image, a numpy array in shape [H,W]
    '''
    hists_syn = np.array(hists_syn)
    hists_ori = np.array(hists_ori)
    H,W = (img_syn.shape[0],img_syn.shape[1])
    num_chosen_filters = len(filter_list)
    print("now the filternums are", num_chosen_filters)
    print(" ---- GIBBS SAMPLING ---- ")
    max_errors = []
    for s in tqdm(range(sweep)):
        for pos_h in tqdm(range(H), desc="handling lines"):
            for pos_w in range(W):
                pos = [pos_h,pos_w]
                img_syn, hists_syn = pos_gibbs_sample_update(img_syn,hists_syn,hists_ori,filter_list,bounds,weight,pos,num_bins,T, my_conv)
        max_error = (np.abs(hists_syn-hists_ori) @ weight).max()
        mean_error = (np.abs(hists_syn-hists_ori) @ weight).mean()
        print(f'Gibbs iteration {s+1}: error = {(np.abs(hists_syn-hists_ori) @ weight).mean()} max_error: {max_error}')
        max_errors.append(max_error)
        # if s % 15 == 0 or s == sweep-1:
        #     plot_histogram(hists_syn[-1], f"results/size64_filter-th_{num_chosen_filters}_syn_histogram_{s}")
        T = T * 0.75
        if max_error < 0.1:
            print(f"Gibbs iteration {s+1}: max_error: {max_error} < 1, sstart_x!")
            break
        if early_stopiing(max_errors):
            print(f"The Gibbs converged at iteration {s+1} with error {max_error}")
            break
    return img_syn, hists_syn.tolist()



def early_stopiing(error_list:list)->bool:
    if len(error_list) < 5:
        return False
    else:
        if error_list[-1] == error_list[-2] == error_list[-3] == error_list[-4]:
            return True
        
        return False


def pos_gibbs_sample_update(img_syn, hists_syn,
                            hists_ori,
                            filter_list, bounds,
                            weight, pos,
                            num_bins, T, my_conv):
    """
    The gibbs sampler for synthesizing a value of single pixel
    """
    H, W = img_syn.shape
    pos_h, pos_w = pos

    # Energy initialization
    histogram_under_values = np.zeros((8, len(filter_list), num_bins))
    distance_for_each_filter = np.zeros((8, len(filter_list)))
    
    # Filtered image pre-computation for efficiency
    filtered_img = np.array([conv(img_syn, f) if my_conv else convolve2d(img_syn, f, mode='same', boundary='wrap')
                             for f in filter_list])

    # Calculate the histograms and distances
    for i in range(8):
        for filter_id in range(len(filter_list)):
            histogram_under_values[i, filter_id, :] = modify_hist(H, W, filtered_img[filter_id], filter_list[filter_id],
                                                                   pos, img_syn[pos_h, pos_w], i,
                                                                   hists_syn[filter_id], num_bins,
                                                                   bounds[filter_id])
            distance_for_each_filter[i, filter_id] = np.sum(cal_norm(histogram_under_values[i, filter_id, :] - hists_ori[filter_id, :], "L2") * weight * H * W)

    # Compute energy and normalize
    energy = distance_for_each_filter.sum(axis=1)
    energy -= energy.min()
    eps = 1e-10
    energy /= (energy.sum() + eps)

    # Probabilities and sampling
    probs = np.exp(-energy / T) + eps
    probs /= probs.sum()
    
    try:
        # Inverse CDF method
        probs_cumsum = np.cumsum(probs)
        rand = random.random()
        index = np.where(probs_cumsum > rand)[0][0]

        # Update image and histogram
        img_syn[pos[0], pos[1]] = index
        hists_syn = histogram_under_values[index, :, :]

    except Exception as e:
        raise ValueError(f'Error in sampling: {e}')

    return img_syn, hists_syn


# a function for norm method
def cal_norm(h, norm):
    if norm == "L1":
        return abs(h)
    elif norm == "L2":
        return h**2 
    else:
        raise ValueError("The norm is not supported!")

# A function to plot the histogram
def plot_histogram(histogram: np.ndarray, title):
    plt.clf()
    plt.plot(histogram, label='histogram')
    plt.legend()
    plt.savefig(title + '.png')

# Modify the histogram based on the intensity change at a single pixel
def modify_hist(img_H, img_W, ori_filtered_img, filter, pos, ori_value, new_value, ori_histogram, bins_num, bounds):
    influenced_patch_on_ori_filtered_img = get_influenced_patch(ori_filtered_img, pos, filter)
    d_value = new_value - ori_value
    delta_response_matrix = np.full((filter.shape[0], filter.shape[1]), d_value)*filter
    new_filtered_img_patch = influenced_patch_on_ori_filtered_img + delta_response_matrix
    
    # use histogram to update
    ori_hist_on_patch = np.histogram(influenced_patch_on_ori_filtered_img, bins=bins_num, range=(bounds[1], bounds[0]))[0]/(img_H*img_W)
    new_hist_on_patch = np.histogram(new_filtered_img_patch, bins=bins_num, range=(bounds[1], bounds[0]))[0]/(img_H*img_W)
    new_histogram = ori_histogram - ori_hist_on_patch + new_hist_on_patch

    return new_histogram

# Get those pixels that will be influenced by the change of the pixel at pos
def get_influenced_patch(ori_image, pos, filter):
    start_y = pos[1]-filter.shape[1] // 2
    end_y = pos[1]+(filter.shape[1]+1) // 2
    start_x = pos[0]-filter.shape[0] // 2
    end_x = pos[0]+(filter.shape[0]+1) // 2
    
    if filter.shape[0] % 2 == 0:
        start_x = start_x + 1
        end_x = end_x + 1
    if filter.shape[1] % 2 == 0:
        start_y = start_y + 1
        end_y = end_y + 1

    if start_y < 0:
        patch = np.hstack((ori_image[:, start_y:], ori_image[:, :end_y]))
    elif end_y > ori_image.shape[1]:
        end_y = end_y - ori_image.shape[1]
        patch = np.hstack((ori_image[:, start_y:], ori_image[:, :end_y]))
    else:
        patch = ori_image[:, start_y:end_y]
    #再在上下方向上切
    if start_x < 0:
        patch = np.vstack((patch[start_x:, :], patch[:end_x, :]))
    elif end_x > ori_image.shape[0]:
        end_x = end_x - ori_image.shape[0]
        patch = np.vstack((patch[start_x:, :], patch[:end_x, :]))
    else:
        patch = patch[start_x:end_x, :]

    return patch


