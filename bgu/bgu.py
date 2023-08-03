import cv2
import sparseqr   # Link: https://github.com/yig/PySPQR
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from .utils import rgb2luminance, gaussian_filter, getWeightMatrix, getSliceMatrix, getOrganisedInputImg, formDxDyDz, apply_Affine_Model_to_HighR_Input, getFinalOutput


def upsampler(highr_input_image, lowr_output_image, gaussian = False, nearest=False, grid_size = [20,15,10,4,3]):
    
    # sh, sw = lowr_output_image.shape[:2]

    highr_input_image = cv2.cvtColor(cv2.imread(highr_input_image), cv2.COLOR_BGR2RGB)
    lowr_output_image = cv2.cvtColor(cv2.imread(lowr_output_image), cv2.COLOR_BGR2RGB)

    tempSz = 320  # downsample size. use a smaller size to speed up Bilateral Guided Upsampling.
    # Note: smaller the size, worse the quality at the same time, larger the size, slower the speed.

    if highr_input_image.shape[0] < highr_input_image.shape[1]:
        lowr_width = tempSz
        lowr_height = highr_input_image.shape[0]/highr_input_image.shape[1] * lowr_width
        lowr_input_image = cv2.resize(highr_input_image, (int(lowr_width), int(lowr_height)))
        lowr_output_image = cv2.resize(lowr_output_image, (int(lowr_width), int(lowr_height)))
    else:
        lowr_height = tempSz
        lowr_width = highr_input_image.shape[1]/highr_input_image.shape[0] * lowr_height
        lowr_input_image = cv2.resize(highr_input_image, (int(lowr_width), int(lowr_height)))
        lowr_output_image = cv2.resize(lowr_output_image, (int(lowr_width), int(lowr_height)))

    highr_input_edge_image = rgb2luminance(highr_input_image)
    highr_input_edge_image = highr_input_edge_image/255.
    lowr_input_edge_image = rgb2luminance(lowr_input_image)
    lowr_input_edge_image = lowr_input_edge_image/255.
    
    if gaussian:
        lowr_input_edge_image = gaussian_filter(lowr_input_edge_image, sigma=0.5, mode='nearest')

    weight_matrix = getWeightMatrix(grid_size,lowr_input_image,lowr_input_edge_image)
    slice_matrix = getSliceMatrix(grid_size,lowr_input_image,lowr_input_edge_image)
    organised_input_img = getOrganisedInputImg(lowr_input_image, grid_size[-1])

    #Prepare b_data
    output_weight = np.ones(lowr_output_image.shape)
    sqrt_w = np.sqrt(output_weight.flatten())
    c1 = lowr_output_image[:,:,0].flatten()
    c2 = lowr_output_image[:,:,1].flatten()
    c3 = lowr_output_image[:,:,2].flatten()
    b_data = np.hstack((c1,c2,c3))
    b_data = np.multiply(b_data,sqrt_w)

    #Prepare a_data
    output_weight_diag_matrix = sparse.diags(sqrt_w)
    A_data = output_weight_diag_matrix * organised_input_img * weight_matrix * slice_matrix

    #Smoothness Term
    [A_deriv_y,A_deriv_x,A_intensity,b_deriv_y,b_deriv_x,b_intensity] = formDxDyDz(lowr_input_image, grid_size)
    b_data = b_data.reshape((b_data.shape[0],1))
    
    #Concat A_data, b_data with smoothness terms
    A = sparse.vstack([A_data,A_deriv_x,A_deriv_y,A_intensity])
    b = sparse.vstack([b_data,b_deriv_x,b_deriv_y,b_intensity])

    #Calculate affine model with smoothness term
    gamma_temp  = sparseqr.solve(A, b,tolerance = 1e-12)
    gamma = np.array(gamma_temp.toarray()).reshape(grid_size)

    #Apply affine model to the high resolution input image
    affine_model = apply_Affine_Model_to_HighR_Input(highr_input_image, highr_input_edge_image, grid_size, gamma, nearest)

    final_output_smooth = getFinalOutput(affine_model, grid_size, highr_input_image)

    return highr_input_image, final_output_smooth, lowr_output_image
