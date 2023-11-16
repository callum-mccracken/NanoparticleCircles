"""Find some circular nanoparticles!"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2 as cv


def read_image(filename: str) -> np.ndarray:
    """Read a TIFF file as grayscale."""
    img = plt.imread(filename)
    img = np.array(Image.open(filename).convert("L"))
    return img

def find_circles(image: np.ndarray) -> list:
    """Return a list of circles found in an image."""
    img = cv.medianBlur(image, 5)  # pylint: disable=maybe-no-member
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT_ALT, 1.5, 20,  # pylint: disable=maybe-no-member
                              param1=300, param2=0.85, minRadius=5, maxRadius=20)
    circles: list = np.around(circles).astype(np.uint16)
    return circles[0]


def main():
    # read in the image
    image = read_image("G_5mms_246d_020.tif")

    # apply circle finder
    circles = find_circles(image)
    
    # make plot with saved circles on top of original grayscale image
    cimg = cv.cvtColor(image, cv.COLOR_GRAY2BGR)  # pylint: disable=maybe-no-member
    for i in circles:
        # draw the outer circle
        cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)  # pylint: disable=maybe-no-member
        # draw the center of the circle
        cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)  # pylint: disable=maybe-no-member

    cv.imwrite('detected_circles.jpg', cimg)  # pylint: disable=maybe-no-member
    print("wrote circles image to detected_circles.jpg")

    # count number of circles
    print(len(circles), "circles found :)")


if __name__ == "__main__":
    main()
