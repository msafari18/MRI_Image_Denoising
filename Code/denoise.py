import nibabel as nib
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_wavelet
import numpy as np
import pywt


def read_image(img_dir1, img_dir2):
    data = nib.load(img_dir1)
    noisy = data.get_fdata()
    data2 = nib.load(img_dir2)
    ground_truth = data2.get_fdata()
    return noisy, ground_truth

def NLM_denoise(img):
    patch_kw = dict(patch_size=2,
                patch_distance=10, # 15 for pd 20 for T1, T2
                multichannel=True)

    sigma_est = np.mean(estimate_sigma(img, multichannel=True))
    # h  = 0.9 for T1, and h = 0.8 for pd and T2
    denoised_img = denoise_nl_means(img, h = 0.7 * sigma_est, fast_mode=True, **patch_kw)

    return denoised_img

def wavelet_denoise(img):

    denoised_img = denoise_wavelet(img, multichannel=True,
                               method='BayesShrink', mode='soft',
                               rescale_sigma=True, wavelet='db1')
    return denoised_img

def plot_image(ground_truth, noisy_img, denoised_img, mode):
    titles = ["original image", "noisy image", "denoised image"]
    fig = plt.figure(figsize=(12, 3))
    images = [ground_truth, noisy_img, denoised_img]
    for i, a in enumerate(images):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.imshow(a[:,100,:], interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    fig.suptitle("wavelet output on "+ mode +" MR image")
    plt.show()


def SNR_cal(ground_truth, img):
    noise = img - ground_truth
    snr = np.mean(ground_truth) / np.mean(noise)
    return snr

def wavelet_transform(img):

    titles = ['Approximation', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(img[90,:,:], 'db2')
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()

noisy, ground_truth = read_image('pd_icbm_normal_1mm_pn9_rf40.mnc', 'pd_icbm_normal_1mm_pn0_rf0.mnc')
denoised_img = wavelet_denoise(noisy)
plot_image(ground_truth, noisy, denoised_img, 'pd')
print("SNR noisy: ",SNR_cal(ground_truth, noisy), "SNR denoised: ",SNR_cal(ground_truth, denoised_img))

