import os
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from model import load_model

def padarray(img, pad):
    if len(img.shape) == 2:
        return np.pad(img, pad, mode='edge')
    else:
        a = np.zeros((img.shape[0]+sum(pad[0]), img.shape[1]+sum(pad[1]), img.shape[2]))
        for c in range(img.shape[-1]):
            a[:,:,c] = np.pad(img[:,:,c], pad, mode='edge')
        return a
    
def laplacian_pyramid(I, nlev=None):
    r, c = I.shape[:2]

    if nlev is None:
        nlev = int(np.floor(np.log2(min(r, c))))

    pyr = [None] * nlev
    f = np.array([0.05, 0.25, 0.4, 0.25, 0.05])
    filt = f[:, np.newaxis] * f[np.newaxis, :]

    J = I.copy()
    for l in range(nlev - 1):
        I = downsample(J, filt)
        odd = 2 * np.array(I.shape) - np.array(J.shape)
        pyr[l] = J - upsample(I, odd, filt)
        J = I

    pyr[nlev - 1] = J
    return pyr

def downsample(I, filt):
    border_mode = 'reflect'

    # Low pass, convolve with separable filter
    R = cv2.filter2D(I, -1, filt, borderType=cv2.BORDER_REFLECT)  # horizontal
    R = cv2.filter2D(R, -1, filt.T, borderType=cv2.BORDER_REFLECT)  # vertical

    # Decimate
    R = R[::2, ::2]
    return R

def upsample(I, odd, filt):
    # Increase resolution
    I = padarray(I, ((1,1),(1,1))) # Pad the image with a 1-pixel border
    r = 2*I.shape[0]
    c = 2*I.shape[1]
    k = I.shape[2]
    R = np.zeros((r, c, k), dtype=I.dtype)
    R[::2, ::2] = 4 * I  # Increase size 2 times; the padding is now 2 pixels wide

    # Interpolate, convolve with separable filter
    R = cv2.filter2D(R, -1, kernel=filt, borderType=cv2.BORDER_CONSTANT)  # horizontal
    R = cv2.filter2D(R, -1, kernel=filt.T, borderType=cv2.BORDER_CONSTANT)  # vertical

    # Remove the border
    R = R[2:r - 2 - odd[0], 2:c - 2 - odd[1]]
    return R


def enhance(input_dir="data/Input_Images/", output_dir="data/Model_Outputs/"):

    model = load_model()

    for f in tqdm(os.listdir(input_dir)):
        I = cv2.cvtColor(cv2.imread(os.path.join(input_dir, f)), cv2.COLOR_BGR2RGB)/255.

        sz = I.shape
        mxdim = max(sz)
        inSz = 512
        L = 4  # number of sub-networks

        if mxdim > inSz:
            S = [1.5, 1.5, 1.5, 1.05]    # scale vector -- tunable hyper-parameter
        else:
            S = [1,1,1,1]

        I_ = I.copy()
        scale_ratio = inSz/max(sz)
        I = cv2.resize(I, (0, 0), fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_CUBIC)

        top_pad, left_pad = [inSz-I.shape[0], inSz-I.shape[1]]
        I = padarray(I, ((top_pad,0),(left_pad,0)))

        pyr = laplacian_pyramid(I, L)

        images = np.zeros((I.shape[0], I.shape[1], I.shape[2]*L))
        for i in range(L):
            images[0:pyr[i].shape[0], 0:pyr[i].shape[1], 3*i:3+3*i] = pyr[i] * S[i]

        input_ = np.expand_dims(images*255, axis=0)
        output_ = model.predict(input_, verbose=False)

        final_out = np.clip(output_[0,:,:,:3], 0, 255).astype(np.uint8)

        final_out = final_out[top_pad:, left_pad:, :]
        
        # plt.imshow(final_out)
        # print(final_out.shape)
        # plt.show()

        cv2.imwrite(os.path.join(output_dir, f), cv2.cvtColor(final_out, cv2.COLOR_RGB2BGR))

    print("Enhanced images saved to: ", output_dir)
