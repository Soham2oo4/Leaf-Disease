#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Removing the color yellow from the image
import cv2
# Load the image
img = cv2.imread("image")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Define a range of colors that represent the yellow
lower_yellow = (20, 100, 100)
upper_yellow = (30, 255, 255)
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
mask = cv2.bitwise_not(mask)
result = cv2.bitwise_and(img, img, mask=mask)
cv2.imwrite("yellow_removed_image", result)


# In[13]:


#Find the percentage of the color yellow in the image
import cv2
img = cv2.imread("image")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_yellow = (20, 100, 100)
upper_yellow = (30, 255, 255)
# Mask that only selects the yellow pixels
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
# Calculate the percentage of yellow pixels
percentage = (cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])) * 100
print("Percentage of yellow color: {:.2f}%".format(percentage))
if percentage > 0.7:
    print("leaf defected")
else:
    print("no defect")


# In[5]:


#Checking if the leaf has brown spots
import cv2
import numpy as np
image = cv2.imread("image")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_brown = np.array([20, 100, 100])
upper_brown = np.array([30, 255, 255])
mask = cv2.inRange(hsv, lower_brown, upper_brown)

# Count the number of brown pixels
brown_pixels = cv2.countNonZero(mask)
# Calculate the percentage of brown pixels
percentage = (cv2.countNonZero(mask) / (image.shape[0] * image.shape[1])) * 100
print("Percentage of brown color: {:.2f}%".format(percentage))
# Check if there are brown spots on the image
if brown_pixels > 0:
    print("There are brown spots on the leaf which implies the leaf has disease.")
else:
    print("There are no brown spots on the leaf which implies the leaf is disease free.")


# In[ ]:


#Checking if leaf is Curled
import cv2
import numpy as np

def is_curled(image):
    # 1. Read and pre-process the image
    img = cv2.imread("image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Use contour detection to find the contours of the leaf
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)

    # 3. Use image moments to calculate the centroid, area, and orientation of the leaf
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        area = cv2.contourArea(c)
        (x, y), (MA, ma), angle = cv2.fitEllipse(c)
    else:
        return "Leaf not found"
    
    # 4. Use the calculated properties to determine whether the leaf is curled up or not
    if angle > 90:
        return "Leaf is curled. If curled it means that there is a chance of disease"
    else:
        return "Leaf is not curled then there will not be a disease"

