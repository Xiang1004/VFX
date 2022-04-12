import math
import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt
import os.path
import sys
from scipy.ndimage import filters

# MATLAB eps
eps = np.spacing (1)


def calculateR (pyramid):

	x_gradient =  np.asarray([[0, 0, 0],[- 0.5, 0, 0.5],[0, 0, 0]]).astype(np.float64)
	y_gradient =  np.asarray([[0, 0.5, 0], [0, 0, 0], [0, - 0.5, 0]]).astype(np.float64)

	dx = cv2.filter2D(pyramid.astype (float), cv2.CV_64F, x_gradient)
	dy = cv2.filter2D(pyramid.astype (float), cv2.CV_64F, y_gradient)
	Iy = cv2.GaussianBlur(dy, (0, 0), sigmaX = 1, sigmaY = 1, borderType = 0)
	Ix = cv2.GaussianBlur(dx, (0, 0), sigmaX = 1, sigmaY = 1, borderType = 0)
	Iyy = Iy * Iy
	Ixx = Ix * Ix
	Ixy = Ix * Iy
	
	Ixx = cv2.GaussianBlur(Ixx, (0, 0), sigmaX = 1.5, sigmaY = 1.5, borderType = 0)
	Iyy = cv2.GaussianBlur(Iyy, (0, 0), sigmaX = 1.5, sigmaY = 1.5, borderType = 0)
	Ixy = cv2.GaussianBlur(Ixy, (0, 0), sigmaX = 1.5, sigmaY = 1.5, borderType = 0)
	
	output = (Ixx * Iyy - np.square (Ixy)) / (Ixx + Iyy + eps)

	return output


def FindLocMax (R, threshold=5):
	TR = R > threshold
	locmax = (TR & (R == cv2.dilate (R, np.ones((3, 3))))) * R

	return locmax

def RemoveBoundary (R, boundary=20):
	R[:boundary, :], R[-boundary:, :], R[:, :boundary], R[:, -boundary:] = 0, 0, 0, 0
	return R

def DescriptorDistance (fpsLoc):
	return np.sqrt (np.sum (np.square (fpsLoc[:, np.newaxis, :] - fpsLoc), axis=2))

def calRadius(h, w):
	r = (h + w + np.sqrt (np.square (h + w) + h * w * 540)) / 359
	return int (r)

def NMS (R, maxFpNum=2000):
	c = 0
	nonZeroIdx = R.nonzero()
	nonZeroCnt = len (nonZeroIdx[-1])
	fpsPos = np.transpose (nonZeroIdx)

	if (nonZeroCnt > maxFpNum):
		fps = R[nonZeroIdx]
		fpsPos = fpsPos[np.argsort (-fps)]
		distance = DescriptorDistance (fpsPos)

		r = calRadius (R.shape[0], R.shape[1])
		fpNum = 0

		while (fpNum < maxFpNum) and not (c > 10):

			suppression = np.zeros_like (fps, dtype=bool)
			for i in range (1, nonZeroCnt):
				suppression[i : ] |= (distance[i - 1, i : ] < r)

			r -= 1
			fpNum = np.count_nonzero (-1 * suppression)
			c += 1

		supFpsIdx = (-1 * suppression).nonzero()
		fpsPos = fpsPos[supFpsIdx[0][ : maxFpNum]]


	return fpsPos

def detectHarrisCorner (imgPyramid):
	fpsPos = NMS(FindLocMax (RemoveBoundary(calculateR (imgPyramid))))
	fps = []
	for i in range (len (fpsPos)):
		x, y = fpsPos[i][1], fpsPos[i][0]
		fps.append ([x, y])
	return fps

def BasePyramid (img, level):
	img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
	img_pyramid = [img]

	for i in range(level):
		img_Guas_temp = cv2.GaussianBlur(img_pyramid[i],(0, 0), sigmaX = 1, sigmaY = 1, borderType = 0)
		img_pyramid.append(cv2.resize (img_Guas_temp, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC))
	return img_pyramid

def LevelHarrisCorner (pyramids):
	features = []
	for level in range (len (pyramids) - 1):
		fps = detectHarrisCorner (pyramids[level])
		features.append(fps)

	return features


def FindFeatures(img, pyrLevel=2):
	pyramids = BasePyramid (img, pyrLevel)
	features = LevelHarrisCorner (pyramids)

	return features

def feature_match(img1, img2):
	
	feature1 = FindFeatures(img1)[0]
	feature2 = FindFeatures(img2)[0]

	r1 = calculateR (BasePyramid(img1, 2)[0])
	r2 = calculateR (BasePyramid(img2, 2)[0])

	lis1 = []
	lis2 = []
	for [i, j] in feature1:
		min_= [float('inf')]
		can = [(0,0)]
		p1 = r1[j-5:j+6, i-5:i+6]

		for [a, b] in feature2:
			p2 = r2[b-5:b+6, a-5:a+6]
			try:
				dis = np.sum((p1-p2)**2)
				min_.append(dis)
				can.append((a,b))
			except:
				pass

		min_sort = sorted(min_)
		try:
			if min_sort[0] < min_sort[1] * 0.8:
				lis1.append([i,j])
				lis2.append(can[min_.index(min(min_))])
		except:
			pass

	return lis1, lis2


def cylindrical_projection(img, focal_length):
	height, width, channel = img.shape
	cylinder_proj = np.zeros(shape=img.shape, dtype=np.uint8)
	'圖片中心為原點算投影'
	for y in range(-int(height/2), int(height/2)):
		for x in range(-int(width/2), int(width/2)):
			cylinder_x = focal_length*math.atan(x/focal_length)
			cylinder_y = focal_length*y/math.sqrt(x**2+focal_length**2)
			'座標補正'
			cylinder_x = round(cylinder_x + width / 2)
			cylinder_y = round(cylinder_y + height / 2)
			'算對應點'
			if cylinder_x >= 0 and cylinder_x < width and cylinder_y >= 0 and cylinder_y < height:
				cylinder_proj[cylinder_y][cylinder_x] = img[y+int(height/2)][x+int(width/2)]
	return cylinder_proj


def fake_ransac(f1, f2):

	ep = 0
	f1 = np.array(f1)
	f2 = np.array(f2)
	
	max_in = 0
	best_trans = [0, 0]

	while ep < 100:
		random = np.random.randint(0, len(f1))
		sample_s = f1[random]
		sample_t = f2[random]

		a = np.array([[1, 0], [0, 1]])
		b = np.array([sample_s[0]-sample_t[0], sample_s[1] - sample_t[1]]).reshape(-1, 1)

		xt, yt = np.linalg.solve(a,b)

		translation = np.array([[xt, yt] for i in range(len(f2))]).reshape(-1, 2).astype(int)
		after_f2 = f2 + translation

		error = np.sum((f1 - after_f2)**2, axis=1)
		in_liner = np.sum(error < 10)
		if in_liner > max_in:
			max_in = in_liner
			best_trans = [xt, yt]
			best_index = error < 10
		
		ep += 1

	f1 = f1[best_index]
	f2 = f2[best_index]

	return f1, f2, best_trans

def image_paste(img1, img2, translation, target_name, mode=1):
	# img1 : 右邊
	# img2 : 左邊

	h1, w1 = img1.shape[:2]
	h2,w2 = img2.shape[:2]
	xt, yt = translation
	xt = int(abs(xt))
	if yt <= 0:
		yt = int(abs(yt))
		
		zero_padding = np.zeros((yt, w1, 3))
		img1 = np.vstack([zero_padding, img1])

		zero_padding = np.zeros((yt, w2, 3))
		img2 = np.vstack([img2, zero_padding])

		new_img = np.zeros((h1+yt, w1+w2, 3))
		
		new_img[5:h2-5, :w2-60] = img2[5:h2-5, :-60]
		new_img[5:h1-5, xt+60:xt+w1] = img1[5:h1-5, 60:]

		new_img = new_img[yt+2:-yt-2, 5: xt+w1]
		if mode == 0:
			ROOT = os.getcwd()
			path = os.path.join(ROOT, 'Result')
			im = cv2.imwrite(os.path.join(path, 'result.jpg'), new_img)
		else:	
			im = cv2.imwrite(os.path.join(input_dirname, target_name+'.jpg'), new_img)

def decide_order(name_list):
	
	start = len(name_list)

	num = 0
	while start != 1:
		
		start = start//2 
		num += start

	result = [str(i+len(name_list))+'.jpg' for i in range(num)]
	
	name_list = [str(i)+'.jpg' for i in range(len(name_list))]	
	return name_list + result
	


if __name__ == '__main__':

	argument = sys.argv[1:][0]
	ROOT = os.getcwd()
	input_dirname = os.path.join(ROOT, argument)

	files = os.listdir(input_dirname)
	files = [i for i in files if i != 'image_list.txt']
	index = len(files)
	output_list = decide_order(files)
	
	for i in range(0, len(output_list), 2):

		a, b = output_list[i], output_list[i+1]
		print("processing "+str(a)+" "+str(b))
		img = cv2.imread(os.path.join(input_dirname, a))
		img1 = cv2.imread(os.path.join(input_dirname, b))

		if i <= 17:
			f = 830
		elif i <= 26:
			f = 1690
		else:
			f = 2490
		warp_img = cylindrical_projection(img, f)
		warp_img1 = cylindrical_projection(img1, f)

		f1, f2 = feature_match(warp_img, warp_img1)

		f1, f2, best_trans = fake_ransac(f1, f2)


		# f1 是左邊
		if np.mean(f1[:, 0]) > np.mean(f2[:, 0]):
			f1, f2 = feature_match(warp_img1, warp_img)
			f1, f2, best_trans = fake_ransac(f1, f2)
			if b == output_list[-1]:
				image_paste(warp_img1, warp_img, best_trans, str(index), mode=0)
			image_paste(warp_img1, warp_img, best_trans, str(index))
			index += 1
		else:
			if b == output_list[-1]:
				image_paste(warp_img, warp_img1, best_trans, str(index), mode=0)
			image_paste(warp_img, warp_img1, best_trans, str(index))
			index += 1
		