import cv2
import numpy as np

# Lower is smoother, which is arguably better.
def metric_smoothness(explanation):
   
	laplacian = cv2.Laplacian(explanation,cv2.CV_64F)
	absolute_laplacian = np.absolute(laplacian)

	return np.average(absolute_laplacian)


volcano_vanilla = cv2.imread("volcano_vanilla.png", cv2.IMREAD_GRAYSCALE)
print("volcano_vanilla: " + str(metric_smoothness(volcano_vanilla)))

volcano_smoothgrad = cv2.imread("volcano_smoothgrad.png", cv2.IMREAD_GRAYSCALE)
print("volcano_smoothgrad: " + str(metric_smoothness(volcano_smoothgrad)))

volcano_xrai = cv2.imread("volcano_xrai.png", cv2.IMREAD_COLOR)
print("volcano_xrai: " + str(metric_smoothness(volcano_xrai)))

volcano_gradcam = cv2.imread("volcano_gradcam.png", cv2.IMREAD_GRAYSCALE)
print("volcano_gradcam: " + str(metric_smoothness(volcano_gradcam)))

volcano_blurIG = cv2.imread("volcano_blurIG.png", cv2.IMREAD_GRAYSCALE)
print("volcano_blurIG: " + str(metric_smoothness(volcano_blurIG)))

volcano_gradcam_smoothed = cv2.imread("volcano_gradcam_smoothed.png", cv2.IMREAD_GRAYSCALE)
print("volcano_gradcam_smoothed: " + str(metric_smoothness(volcano_gradcam_smoothed)))

volcano_blurIG_smoothed = cv2.imread("volcano_blurIG_smoothed.png", cv2.IMREAD_GRAYSCALE)
print("volcano_blurIG_smoothed: " + str(metric_smoothness(volcano_blurIG_smoothed)))
