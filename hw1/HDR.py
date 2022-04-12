import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse


def sample_intensities(images):
    num_images = len(images)
    row = len(images[0]) - 1
    col = len(images[0][0]) - 1
    sample = np.zeros((256, num_images), dtype=np.uint8)     # (256, 7)

    for i in range(0, 256):
        x = random.randint(0, row)
        y = random.randint(0, col)
        for j in range(num_images):
            sample[i, j] = images[j][x, y]
    return sample


def response_curve(Z, B, l, w):
    n = 256
    A = np.zeros(shape=(Z.shape[0] * Z.shape[1] + n + 1, n + Z.shape[0]))
    b = np.zeros(shape=(A.shape[0], 1))
    k = 0

    for i in range(np.size(Z, 0)):
        for j in range(np.size(Z, 1)):
            z = Z[i][j]
            wij = w[z]
            A[k][z] = wij
            A[k][n + i] = -wij
            b[k][0] = wij * B[j]
            k += 1
    A[k][128] = 1
    k += 1

    for i in range(n - 1):
        A[k][i] = l * w[i + 1]
        A[k][i + 1] = -2 * l * w[i + 1]
        A[k][i + 2] = l * w[i + 1]
        k += 1
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    g = x[:256]

    return g


def construct_hdr(img_list, g_function, exposure_times, w):
    img_size = img_list[0][0].shape       # (1080, 1920)

    # HDR radiance map
    hdr = np.zeros((img_size[0], img_size[1], 3), dtype=np.float32)

    for i in range(3):
        Z = [img.flatten().tolist() for img in img_list[i]]
        lE = reconstructed_radiance_map(g_function[i], Z, np.array(exposure_times), w)
        hdr[:, :, i] = np.reshape(np.exp(lE), img_size)     # Exponential each channels and reshape to 2D-matrix

    return hdr


def reconstructed_radiance_map(g, Z, exposure_times, w):
    num = [0] * len(Z[0])
    lE = [0] * len(Z[0])
    pixels, imgs = len(Z[0]), len(Z)
    for i in range(pixels):
        sum_w = 0
        for j in range(imgs):
            z = Z[j][i]
            num[i] += w[z] * (g[z] - exposure_times[j])
            sum_w += w[z]
        if sum_w > 0:
            lE[i] = num[i] / sum_w
        else:
            lE[i] = num[i]
    return lE


def tone_mapping(radiance, a=0.2):

    gray_radiance = cv2.cvtColor(radiance, cv2.COLOR_BGR2GRAY)
    Lw_avg = np.exp(np.mean(np.log(gray_radiance)))
    Lm = a * radiance / Lw_avg

    result = [cv2.GaussianBlur(Lm, (1, 1), 0, 0)]
    fi = 8

    for i in range(4):
        s = i*2 + 3
        result.append(cv2.GaussianBlur(Lm, (s, s), 0, 0))
        vs = (result[i] - result[i + 1]) / (np.sum(result[i]) + ((2 ** fi) * a) / ((2 * s + 1) ** 2))
        if np.all(vs < 0.05):
            max_result = result[i]
    ld = Lm / (1 + max_result)
    LDR = np.clip(np.rint(ld * 255), 0, 255).astype(np.uint8)

    return LDR


def main(exposure_time, smoothness_lambda):
    parser = argparse.ArgumentParser(description='main function of high dynamic range')
    parser.add_argument('--image_path', default='./image', help='path to input image')
    parser.add_argument('--output_path', default='./output', help='path to setting file')
    args = parser.parse_args()

    # images shape (7, 1080, 1920, 3)
    images = np.vstack([np.expand_dims(np.array(cv2.imread(os.path.join(args.image_path, img))), axis=0)
                        for img in os.listdir(os.path.join(args.image_path))])
    # parameter
    exposure_times = [np.log2(e) for e in exposure_time]
    w = [i for i in range(int(256 / 2))] + [i for i in reversed(range(int(256 / 2)))]

    # Calculation response function
    g_function = []
    for BGR in range(3):
        BGR_img = [img[:, :, BGR] for img in images]
        intensity_sampling = sample_intensities(BGR_img)
        Z = np.array([img.flatten() for img in intensity_sampling])
        g = response_curve(Z, exposure_times, smoothness_lambda, w)
        g_function.append(g)
    g_function = np.array(g_function)

    # Plot response curve
    plt.figure(figsize=(10, 10))
    plt.plot(g_function[0], range(256), 'b--', linewidth=4)
    plt.plot(g_function[1], range(256), 'g--', linewidth=4)
    plt.plot(g_function[2], range(256), 'r--', linewidth=4)
    plt.title('response curve')
    plt.ylabel('pixel value Z')
    plt.xlabel('log exposure X')
    plt.savefig(os.path.join(args.output_path, 'response_curve.png'))

    # Construct HDR image
    img_b, img_g, img_r = images[:, :, :, 0], images[:, :, :, 1], images[:, :, :, 2]
    hdr_image = construct_hdr([img_b, img_g, img_r], g_function, exposure_times, w)
    cv2.imwrite(os.path.join(args.output_path, 'HDR.hdr'), hdr_image)

    # Reconstructed radiance map
    plt.figure(figsize=(24, 24))
    plt.imshow(np.log(cv2.cvtColor(hdr_image, cv2.COLOR_BGR2GRAY)), cmap='jet')
    plt.savefig(os.path.join(args.output_path, 'radiance_map.png'))

    # Tone mapping
    tonemapping = tone_mapping(hdr_image, 0.5)
    cv2.imwrite(os.path.join(args.output_path, 'tone_mapping.png'), tonemapping)


if __name__ == '__main__':
    exposure_time = [1/15, 1/30, 1/60, 1/125, 1/250, 1/500, 1/1000]
    smoothness_lambda = 100
    main(exposure_time, smoothness_lambda)
