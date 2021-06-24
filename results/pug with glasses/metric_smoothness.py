import cv2
import numpy as np

# Lower is smoother, which is arguably better.
def metric_smoothness(explanation):
   
	laplacian = cv2.Laplacian(explanation,cv2.CV_64F)
	absolute_laplacian = np.absolute(laplacian)

	return np.average(absolute_laplacian)


pug_vanilla = cv2.imread("pugg_vanilla.png", cv2.IMREAD_GRAYSCALE)
print("pug_vanilla: " + str(metric_smoothness(pug_vanilla)))

pug_smoothgrad = cv2.imread("pugg_smoothgrad.png", cv2.IMREAD_GRAYSCALE)
print("pug_smoothgrad: " + str(metric_smoothness(pug_smoothgrad)))

pug_xrai = cv2.imread("pugg_xrai.png", cv2.IMREAD_COLOR)
print("pug_xrai: " + str(metric_smoothness(pug_xrai)))

pug_gradcam = cv2.imread("pugg_gradcam.png", cv2.IMREAD_GRAYSCALE)
print("pug_gradcam: " + str(metric_smoothness(pug_gradcam)))

pug_blurIG = cv2.imread("pugg_blurIG.png", cv2.IMREAD_GRAYSCALE)
print("pug_blurIG: " + str(metric_smoothness(pug_blurIG)))

pug_gradcam_smoothed = cv2.imread("pugg_gradcam_smoothed.png", cv2.IMREAD_GRAYSCALE)
print("pug_gradcam_smoothed: " + str(metric_smoothness(pug_gradcam_smoothed)))

pug_blurIG_smoothed = cv2.imread("pugg_blurIG_smoothed.png", cv2.IMREAD_GRAYSCALE)
print("pug_blurIG_smoothed: " + str(metric_smoothness(pug_blurIG_smoothed)))