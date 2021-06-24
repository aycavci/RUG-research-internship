import cv2
import numpy as np

# Lower is smoother, which is arguably better.
def metric_smoothness(explanation):
   
	laplacian = cv2.Laplacian(explanation,cv2.CV_64F)
	absolute_laplacian = np.absolute(laplacian)

	return np.average(absolute_laplacian)


doberman_vanilla = cv2.imread("doberman_vanilla.png", cv2.IMREAD_GRAYSCALE)
print("doberman_vanilla: " + str(metric_smoothness(doberman_vanilla)))

doberman_smoothgrad = cv2.imread("doberman_smoothgrad.png", cv2.IMREAD_GRAYSCALE)
print("doberman_smoothgrad: " + str(metric_smoothness(doberman_smoothgrad)))

doberman_xrai = cv2.imread("doberman_xrai.png", cv2.IMREAD_COLOR)
print("doberman_xrai: " + str(metric_smoothness(doberman_xrai)))

doberman_gradcam = cv2.imread("doberman_gradcam.png", cv2.IMREAD_GRAYSCALE)
print("doberman_gradcam: " + str(metric_smoothness(doberman_gradcam)))

doberman_blurIG = cv2.imread("doberman_blurIG.png", cv2.IMREAD_GRAYSCALE)
print("doberman_blurIG: " + str(metric_smoothness(doberman_blurIG)))

doberman_gradcam_smoothed = cv2.imread("doberman_gradcam_smoothed.png", cv2.IMREAD_GRAYSCALE)
print("doberman_gradcam_smoothed: " + str(metric_smoothness(doberman_gradcam_smoothed)))

doberman_blurIG_smoothed = cv2.imread("doberman_blurIG_smoothed.png", cv2.IMREAD_GRAYSCALE)
print("doberman_blurIG_smoothed: " + str(metric_smoothness(doberman_blurIG_smoothed)))