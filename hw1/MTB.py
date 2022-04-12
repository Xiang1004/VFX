import cv2
import numpy as np
import sys


def Bitmap_Binary(img):
    exb_array = img.copy()
    bin_array = img.copy()
    threshold = np.median(img)
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            exb_array[row][column] = 1 - (threshold - 10 < img[row][column] < threshold + 10)
            bin_array[row][column] = 1 - (img[row][column] < threshold)
    return exb_array, bin_array


def shift(img, shift_x, shift_y):
    img_array = img.copy()
    for row in range(img_array.shape[0]):
        for column in range(img_array.shape[1]):
            if img_array.shape[0] - 1 > row + shift_x >= 0 and img_array.shape[1] - 1 > column + shift_y >= 0:
                img_array[row][column] = img[row + shift_x][column + shift_y]
    return img_array


s_img = cv2.imread('01.jpg', 0)
m_img = cv2.imread('02.jpg', 0)

shift_img = cv2.imread('01.jpg')

s_Bitmap, s_Binary = Bitmap_Binary(s_img)
m_Bitmap, m_Binary = Bitmap_Binary(m_img)

s_reBinary = []
m_reBinary = []
s_reBitmap = []
m_reBitmap = []

for i in range(1, 6):
    s_reBinary.append(
        cv2.resize(s_Binary, (s_Binary.shape[0] // 2 ** i, s_Binary.shape[1] // 2 ** i), interpolation=cv2.INTER_CUBIC))
    m_reBinary.append(
        cv2.resize(m_Binary, (m_Binary.shape[0] // 2 ** i, m_Binary.shape[1] // 2 ** i), interpolation=cv2.INTER_CUBIC))
    s_reBitmap.append(
        cv2.resize(s_Bitmap, (s_Bitmap.shape[0] // 2 ** i, s_Bitmap.shape[1] // 2 ** i), interpolation=cv2.INTER_CUBIC))
    m_reBitmap.append(
        cv2.resize(m_Bitmap, (m_Bitmap.shape[0] // 2 ** i, m_Bitmap.shape[1] // 2 ** i), interpolation=cv2.INTER_CUBIC))

shift_x = 0
shift_y = 0
result = []

for num in range(5):
    maxerror = sys.maxsize
    move_x = int(shift_x)
    move_y = int(shift_y)
    Bin_mask = shift(s_reBinary[4 - num], move_x, move_y)
    Exb_mask = shift(s_reBitmap[4 - num], move_x, move_y)
    max_i = 0
    max_j = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            shiftBin = shift(Bin_mask, i, j)
            shiftExb = shift(Exb_mask, i, j)
            XOR = cv2.bitwise_xor(shiftBin, m_reBinary[4 - num])
            AND_1 = cv2.bitwise_and(XOR, m_reBitmap[4 - num])
            AND_2 = cv2.bitwise_and(AND_1, shiftExb)
            if maxerror > (np.sum(AND_2)):
                maxerror = np.sum(AND_2)
                max_i = i
                max_j = j
    shift_x = shift_x * 2 + max_i * 2
    shift_y = shift_y * 2 + max_j * 2
    print(max_i, max_j)

output = shift(shift_img, int(shift_x), int(shift_y))
cv2.imwrite('15.jpg', output)   # 15、30、60、125、250、500、1000
