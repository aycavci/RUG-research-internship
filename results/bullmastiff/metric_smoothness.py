import cv2
import numpy as np

# Lower is smoother, which is arguably better.
def metric_smoothness(explanation):
   
	laplacian = cv2.Laplacian(explanation,cv2.CV_64F)
	absolute_laplacian = np.absolute(laplacian)

	return np.average(absolute_laplacian)


bull_vanilla = cv2.imread("bull_vanilla.png", cv2.IMREAD_GRAYSCALE)
print("bull_vanilla: " + str(metric_smoothness(bull_vanilla)))

bull_smoothgrad = cv2.imread("bull_smoothgrad.png", cv2.IMREAD_GRAYSCALE)
print("bull_smoothgrad: " + str(metric_smoothness(bull_smoothgrad)))

bull_xrai = cv2.imread("bull_xrai.png", cv2.IMREAD_COLOR)
print("bull_xrai: " + str(metric_smoothness(bull_xrai)))

bull_gradcam = cv2.imread("bull_gradcam.png", cv2.IMREAD_GRAYSCALE)
print("bull_gradcam: " + str(metric_smoothness(bull_gradcam)))

bull_blurIG = cv2.imread("bull_blurIG.png", cv2.IMREAD_GRAYSCALE)
print("bull_blurIG: " + str(metric_smoothness(bull_blurIG)))

bull_gradcam_smoothed = cv2.imread("bull_gradcam_smoothed.png", cv2.IMREAD_GRAYSCALE)
print("bull_gradcam_smoothed: " + str(metric_smoothness(bull_gradcam_smoothed)))

bull_blurIG_smoothed = cv2.imread("bull_blurIG_smoothed.png", cv2.IMREAD_GRAYSCALE)
print("bull_blurIG_smoothed: " + str(metric_smoothness(bull_blurIG_smoothed)))
