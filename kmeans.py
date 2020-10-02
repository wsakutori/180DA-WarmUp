# Libraries from OpenCV python samples

#!/usr/bin/env python

'''
K-means clusterization sample.
Usage:
   kmeans.py
Keyboard shortcuts:
   ESC   - exit
   space - generate new distribution
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 
import sys

def main():
    cluster_n = 3
    img_size = 512

    # generating bright palette
    colors = np.zeros((1, cluster_n, 3), np.uint8)
    colors[0,:] = 255
    colors[0,:,0] = np.arange(0, 180, 180.0/cluster_n)
    colors = cv.cvtColor(colors, cv.COLOR_HSV2BGR)[0]

    cap = cv.VideoCapture(0)

    while True:
        print('sampling webcam...')

        # Capture frame-by-frame
        _, img = cap.read()

        pixel_values = img.reshape((-1,3))
        pixel_values = np.float32(pixel_values)

        #term_crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)
        _, labels, (centers) = cv.kmeans(pixel_values, cluster_n, None, term_crit, 10, cv.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)

        seg_img = centers[labels.flatten()]

        seg_img = seg_img.reshape(img.shape)

        labels = labels.reshape(img.shape[0], img.shape[1])

        cv.imshow("seg_img", seg_img)
        if cv.waitKey(1) == ord("q"):
            break   

    print('Done')
    # When everything done, release the capture

if __name__ == '__main__':
    print(__doc__)
    main()
    cap.release()
    cv.destroyAllWindows()