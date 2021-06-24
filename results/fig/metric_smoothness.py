import cv2
import numpy as np

# Lower is smoother, which is arguably better.
def metric_smoothness(explanation):
   
	laplacian = cv2.Laplacian(explanation,cv2.CV_64F)
	absolute_laplacian = np.absolute(laplacian)

	return np.average(absolute_laplacian)


fig_vanilla = cv2.imread("fig_vanilla.png", cv2.IMREAD_GRAYSCALE)
print("fig_vanilla: " + str(metric_smoothness(fig_vanilla)))

fig_smoothgrad = cv2.imread("fig_smoothgrad.png", cv2.IMREAD_GRAYSCALE)
print("fig_smoothgrad: " + str(metric_smoothness(fig_smoothgrad)))

fig_xrai = cv2.imread("fig_xrai.png", cv2.IMREAD_COLOR)
print("fig_xrai: " + str(metric_smoothness(fig_xrai)))

fig_gradcam = cv2.imread("fig_gradcam.png", cv2.IMREAD_GRAYSCALE)
print("fig_gradcam: " + str(metric_smoothness(fig_gradcam)))

fig_blurIG = cv2.imread("fig_blurIG.png", cv2.IMREAD_GRAYSCALE)
print("fig_blurIG: " + str(metric_smoothness(fig_blurIG)))

fig_gradcam_smoothed = cv2.imread("fig_gradcam_smoothed.png", cv2.IMREAD_GRAYSCALE)
print("fig_gradcam_smoothed: " + str(metric_smoothness(fig_gradcam_smoothed)))

fig_blurIG_smoothed = cv2.imread("fig_blurIG_smoothed.png", cv2.IMREAD_GRAYSCALE)
print("fig_blurIG_smoothed: " + str(metric_smoothness(fig_blurIG_smoothed)))
