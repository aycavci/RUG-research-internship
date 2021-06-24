import cv2
import numpy as np

# Lower is smoother, which is arguably better.
def metric_smoothness(explanation):
   
	laplacian = cv2.Laplacian(explanation,cv2.CV_64F)
	absolute_laplacian = np.absolute(laplacian)

	return np.average(absolute_laplacian)


fish_vanilla = cv2.imread("fish_vanilla.png", cv2.IMREAD_GRAYSCALE)
print("fish_vanilla: " + str(metric_smoothness(fish_vanilla)))

fish_smoothgrad = cv2.imread("fish_smoothgrad.png", cv2.IMREAD_GRAYSCALE)
print("fish_smoothgrad: " + str(metric_smoothness(fish_smoothgrad)))

fish_xrai = cv2.imread("fish_xrai.png", cv2.IMREAD_COLOR)
print("fish_xrai: " + str(metric_smoothness(fish_xrai)))

fish_gradcam = cv2.imread("fish_gradcam.png", cv2.IMREAD_GRAYSCALE)
print("fish_gradcam: " + str(metric_smoothness(fish_gradcam)))

fish_blurIG = cv2.imread("fish_blurIG.png", cv2.IMREAD_GRAYSCALE)
print("fish_blurIG: " + str(metric_smoothness(fish_blurIG)))

fish_gradcam_smoothed = cv2.imread("fish_gradcam_smoothed.png", cv2.IMREAD_GRAYSCALE)
print("fish_gradcam_smoothed: " + str(metric_smoothness(fish_gradcam_smoothed)))

fish_blurIG_smoothed = cv2.imread("fish_blurIG_smoothed.png", cv2.IMREAD_GRAYSCALE)
print("fish_blurIG_smoothed: " + str(metric_smoothness(fish_blurIG_smoothed)))
