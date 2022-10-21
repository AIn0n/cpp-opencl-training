import numpy as np
import cv2

arr = np.array([
    [0, 11, 11, 3, 0, 0],
    [3, 11, 3, 11, 0, 0],
    [0, 11, 3, 3, 0, 0],
    [0, 11, 11, 11, 0, 0],
    [0, 11, 11, 11, 0, 0],
    [0, 11, 11, 11, 0, 0],
    [0, 11, 11, 11, 0, 0]
])

output = cv2.medianBlur(np.int16(arr), 5)
print(output)