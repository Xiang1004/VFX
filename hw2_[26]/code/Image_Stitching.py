import numpy as np
import cv2
import os
import argparse

eps = 1e-5


def cylindrical_projection(img, focal_length):
    h, w = img.shape[:2]
    cylinder_proj = np.zeros(shape=img.shape, dtype=np.uint8)

    for y in range(-int(h / 2), int(h / 2)):
        for x in range(-int(w / 2), int(w / 2)):
            cylinder_x = focal_length * np.arctan(x / focal_length)
            cylinder_y = focal_length * y / np.sqrt(x ** 2 + focal_length ** 2)

            cylinder_x = round(cylinder_x + w / 2)
            cylinder_y = round(cylinder_y + h / 2)

            if 0 <= cylinder_x < w and 0 <= cylinder_y < h:
                cylinder_proj[cylinder_y][cylinder_x] = img[y + int(h / 2)][x + int(w / 2)]
    return cylinder_proj


def calculateR(pyramid):
    x_gradient = np.asarray([[0, 0, 0], [- 0.5, 0, 0.5], [0, 0, 0]]).astype(np.float64)
    y_gradient = np.asarray([[0, 0.5, 0], [0, 0, 0], [0, - 0.5, 0]]).astype(np.float64)

    dx = cv2.filter2D(pyramid.astype(float), cv2.CV_64F, x_gradient)
    dy = cv2.filter2D(pyramid.astype(float), cv2.CV_64F, y_gradient)
    Iy = cv2.GaussianBlur(dy, (0, 0), sigmaX=1, sigmaY=1, borderType=0)
    Ix = cv2.GaussianBlur(dx, (0, 0), sigmaX=1, sigmaY=1, borderType=0)

    Ixx = cv2.GaussianBlur(Ix * Ix, (0, 0), sigmaX=1.5, sigmaY=1.5, borderType=0)
    Iyy = cv2.GaussianBlur(Iy * Iy, (0, 0), sigmaX=1.5, sigmaY=1.5, borderType=0)
    Ixy = cv2.GaussianBlur(Ix * Iy, (0, 0), sigmaX=1.5, sigmaY=1.5, borderType=0)

    output = (Ixx * Iyy - np.square(Ixy)) / (Ixx + Iyy + eps)

    return output


def detectHarrisCorner(imgPyramid):
    fpsPos = NMS(FindLocMax(imgPyramid))
    fps = []
    for i in range(len(fpsPos)):
        x, y = fpsPos[i][1], fpsPos[i][0]
        fps.append([x, y])
    return fps


def NMS(R, maxFpNum=2000):
    c = 0
    h, w = R.shape[:2]
    img_h, img_w = image_h_w

    nonZeroIdx = R.nonzero()
    nonZeroCnt = len(nonZeroIdx[-1])
    fpsPos = np.transpose(nonZeroIdx)

    if nonZeroCnt > maxFpNum:
        fps = R[nonZeroIdx]
        fpsPos = fpsPos[np.argsort(-fps)]
        distance = np.sqrt(np.sum(np.square(fpsPos[:, np.newaxis, :] - fpsPos), axis=2))  # Descriptor Distance

        r = int((h + w + np.sqrt(np.square(h + w) + h * w * img_h)) / img_w)
        fpNum = 0

        while (fpNum < maxFpNum) and not (c > 8):

            suppression = np.zeros_like(fps, dtype=bool)
            for i in range(1, nonZeroCnt):
                suppression[i:] |= (distance[i - 1, i:] < r)

            r -= 1
            fpNum = np.count_nonzero(-1 * suppression)
            c += 1

        supFpsIdx = (-1 * suppression).nonzero()
        fpsPos = fpsPos[supFpsIdx[0][: maxFpNum]]

    return fpsPos


def FindLocMax(R, boundary=20, threshold=5):
    R[:boundary, :], R[-boundary:, :], R[:, :boundary], R[:, -boundary:] = 0, 0, 0, 0  # Remove Boundary

    TR = R > threshold
    locmax = (TR & (R == cv2.dilate(R, np.ones((3, 3))))) * R

    return locmax


def feature_match(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    r1 = calculateR(img1)
    r2 = calculateR(img2)

    feature1 = detectHarrisCorner(r1)
    feature2 = detectHarrisCorner(r2)

    f1, f2 = [], []
    for [i, j] in feature1:
        min_ = [float('inf')]
        can = [(0, 0)]
        p1 = r1[j - 5:j + 6, i - 5:i + 6]

        for [a, b] in feature2:
            p2 = r2[b - 5:b + 6, a - 5:a + 6]

            dis = np.sum((p1 - p2) ** 2)
            min_.append(dis)
            can.append((a, b))

        min_sort = sorted(min_)

        if min_sort[0] < min_sort[1] * 0.8:
            f1.append([i, j])
            f2.append(can[min_.index(min(min_))])
    return np.array(f1), np.array(f2)


def fake_ransac(f1, f2):
    max_in = 0
    best_trans = [0, 0]

    for i in range(100):
        random = np.random.randint(0, len(f1))

        x = np.array([[1, 0], [0, 1]])
        y = np.array([f1[random][0] - f2[random][0], f1[random][1] - f2[random][1]]).reshape(-1, 1)

        xt, yt = np.linalg.solve(x, y)

        translation = np.array([[xt, yt] for i in range(len(f2))]).reshape(-1, 2).astype(int)
        after_f2 = f2 + translation

        error = np.sum((f1 - after_f2) ** 2, axis=1)
        in_liner = np.sum(error < 10)
        if in_liner > max_in:
            max_in = in_liner
            best_trans = [xt, yt]
            best_index = error < 10

    f1 = f1[best_index]
    f2 = f2[best_index]

    return f1, f2, best_trans


def image_paste(img1, img2, translation, img_name):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    xt, yt = translation

    if yt <= 100:
        xt, yt = int(abs(xt)), int(abs(yt))

        zero_padding = np.zeros((yt, w1, 3))
        img1 = np.vstack([zero_padding, img1])

        zero_padding = np.zeros((yt, w2, 3))
        img2 = np.vstack([img2, zero_padding])

        result = np.zeros((h1 + yt, w1 + w2, 3))

        result[5:h2 - 5, :w2 - 60] = img2[5:h2 - 5, :-60]
        result[5:h1 - 5, xt + 60:xt + w1] = img1[5:h1 - 5, 60:]

        result = result[yt + 2:-yt - 2, 5: xt + w1]

        cv2.imwrite(os.path.join(images_path, img_name + '.jpg'), result)


def decide_order(num_image):
    stitching = [str(i + num_image) + '.jpg' for i in range(num_image - 2)]
    name_list = [str(i) + '.jpg' for i in range(num_image)]

    return name_list + stitching


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='images_path', type=str, default='./data')
    parser.add_argument('--focal', help='focal_length', type=int, default=2350)
    args = parser.parse_args()

    images_path = os.path.join(args.path)
    files = os.listdir(images_path)
    name = index = len(files)
    output_list = decide_order(index)

    for i in range(0, len(output_list), 2):

        pic1, pic2 = output_list[i], output_list[i + 1]
        print("Stitching " + str(pic1) + " " + str(pic2))

        img1 = cv2.imread(os.path.join(images_path, pic1))
        img2 = cv2.imread(os.path.join(images_path, pic2))
        image_h_w = img1.shape[:2]

        if i <= index - 1:
            focal = args.focal
        elif i <= index - 1 + (index // 2):
            focal = 2 * args.focal
        else:
            focal = 3 * args.focal

        warp_img1 = cylindrical_projection(img1, focal)
        warp_img2 = cylindrical_projection(img2, focal)

        f1, f2 = feature_match(warp_img1, warp_img2)
        f1, f2, best_trans = fake_ransac(f1, f2)

        if np.mean(f1[:, 0]) < np.mean(f2[:, 0]):
            image_paste(warp_img1, warp_img2, best_trans, str(name))
        else:
            f1, f2 = feature_match(warp_img2, warp_img1)
            f1, f2, best_trans = fake_ransac(f1, f2)
            image_paste(warp_img2, warp_img1, best_trans, str(name))
        name += 1
