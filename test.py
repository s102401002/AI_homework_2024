import cv2

image = cv2.imread('hw3/image/0.png', cv2.IMREAD_GRAYSCALE)
image = image.resize((28, 28))
print(image)