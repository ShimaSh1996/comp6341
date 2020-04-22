#!/usr/bin/env python
# coding: utf-8

# # Computer Vision 

# ## Assignment 2

# In[114]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import cv2
from matplotlib import pyplot as plt
from operator import itemgetter


# In[115]:


def Compute_GRAD(image):
    kernel_x = (1/8) * np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
    kernel_y = (1/8) * np.array([[1, 2, 1],
                                 [0, 0, 0],
                                 [-1, -2, -1]])
    grad_x = cv2.filter2D(image, -1, kernel_x)
    grad_y = cv2.filter2D(image, -1, kernel_y)
    return grad_x, grad_y


# In[116]:


path = 'image_sets/yosemite/Yosemite1.jpg'
img = cv2.imread(path, 0)/255
real_img = cv2.imread(path)


# In[117]:


path = 'image_sets/yosemite/Yosemite2.jpg'
img_second = cv2.imread(path, 0)/255
real_img_second = cv2.imread(path)


# In[24]:


# path = 'image_sets/graf/img1.ppm'
# img = cv2.imread(path, 0)/255
# real_img = cv2.imread(path)


# In[25]:


# path = 'image_sets/graf/img2.ppm'
# img_second = cv2.imread(path, 0)/255
# real_img_second = cv2.imread(path)


# In[118]:


plt.imshow(img)


# In[119]:


# print(img.shape)


# ### Feature Detection

# In[120]:


def Compute_HARRIS_MAT(grad_x, grad_y):
    grad_x2 = grad_x**2
    grad_y2 = grad_y**2
    grad_x2 = cv2.GaussianBlur(grad_x2, (5, 5), 3, 3)
    grad_y = cv2.GaussianBlur(grad_y2, (5, 5), 3, 3)
    grad_xy = cv2.GaussianBlur(grad_x*grad_y , (5, 5), 3, 3)
    return grad_x2, grad_y2, grad_xy

def Create_H(grad_x2, grad_y2, grad_xy):
    upper_left = cv2.boxFilter(grad_x2, ksize=(5,5), ddepth= -1, normalize = False)
    upper_right = cv2.boxFilter(grad_xy, ksize=(5,5), ddepth= -1, normalize = False)
#     lower_left = cv2.boxFilter(grad_xy, ksize=(5,5), ddepth= -1, normalize = False)
    lower_right = cv2.boxFilter(grad_y2, ksize=(5,5), ddepth=-1,normalize = False)
    return upper_left ,lower_right, upper_right
    
def Corner_Response(grad_x2, grad_y2, grad_xy, alpha = 0.04):
    return ((grad_x2*grad_y2 - grad_xy*grad_xy) - alpha*(grad_x2+grad_y2))

def Threshold(array: np.ndarray, thresh: float):
    thresh_array = np.copy(array)
    thresh_array[array < thresh] = 0
    return thresh_array

def Non_Max_Supression(array: np.ndarray, kernel_size: int = 5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(array, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=[0, 0, 0])
    max_array = np.copy(array)
    max_array[max_array < dilated] = 0
    return max_array


# In[121]:


def Computate_Coner_Strength(img):
    I_x, I_y = Compute_GRAD(img)
    I_x2, I_y2, I_xy = Compute_HARRIS_MAT(I_x, I_y)
    H_x, H_y, H_xy = Create_H(I_x2, I_y2, I_xy)
    R = Corner_Response(H_x, H_y, H_xy)
    thresh = Threshold(R, 0.02)
#     print(np.sum(thresh))
    res = Non_Max_Supression(thresh, 5)
    return res


# In[122]:


result = Computate_Coner_Strength(img)
result_second = Computate_Coner_Strength(img_second)

# cv2.imshow("test", result_second*255)
# cv2.waitKey(0)
# cv2.imshow("result", result*255)
# cv2.waitKey(0)
print(result.shape)


# ### Feature Description

# In this section I implemented SIFT Descriptor, Rotation Invariant, and Scale Invariant.

# In[123]:


def Extract_Keypoints(corner_strength):        
    keypoints = []
    for row in range(corner_strength.shape[0]):
        for column in range(corner_strength.shape[1]):
            if corner_strength[row][column]>0 :
#             print(result[row][column])
                keypoint = corner_strength[row][column]
                keypoints.append([row,column,keypoint])
    # keypoints = np.array(keypoints)
    # print(keypoints)
    return keypoints

def Feature_Description(img, keypoints):
    dx, dy = Compute_GRAD(img)
    magnitude, angle = np.zeros_like(dx), np.zeros_like(dx)
    magnitude, angle = cv2.cartToPolar(dx, dy, magnitude, angle, angleInDegrees=True)
    bins = np.linspace(0, 360, 9, endpoint=True)
    
    patches = []
    left, top, bottom, right = 7, 7, 8, 8
    
    descriptors = []
    for index in range(len(keypoints)):
        row = keypoints[index][0] + left
        col = keypoints[index][1] + top
        image = np.array(img[row - top: row + bottom + 1, col - left: col + right + 1])
        magnitude = np.array(magnitude[row - top: row + bottom + 1,
                                        col - left: col + right + 1])
        orientation = np.array(angle[row - top: row + bottom + 1,
                                      col - left: col + right + 1])
        descriptors_i = np.zeros((16,8))
        rotation = np.zeros((1,8))
#         print(rotation.shape)
        cnt = 0
        for i in range(0,16,4):
            for j in range(0,16,4):
                test = np.array(orientation[i:i+4, j:j+4])
                values, angles = np.histogram(a=test, bins=bins)
                descriptors_i[cnt] = values/16
#                 print(descriptors_i)
                cnt +=1
        dominant_rotation = np.argmax(np.sum(descriptors_i, axis=0))
        if dominant_rotation != 0:
            descriptors_i = np.roll(descriptors_i, -dominant_rotation, axis=1)
#         print("new_desc:", descriptors_i)
        temp = np.clip(descriptors_i, a_min=0.0, a_max=0.2)
        contrast_invariant = temp**2/ np.sum(temp**2)
#         print(np.sum(contrast_invariant))
        descriptors.append([keypoints[index][0],keypoints[index][1],contrast_invariant])
    return descriptors


# In[124]:


keypoints = Extract_Keypoints(result)
patches = Feature_Description(img, keypoints)
keypoints_second = Extract_Keypoints(result_second)
patches_second = Feature_Description(img_second, keypoints_second)


# In[125]:


# print(patches[15][1])
# print(len(keypoints))


# ### Feature matching

# In[126]:


def SSD(feature_1, feature_2):
    distance = np.subtract(feature_1, feature_2)
    return float(np.sum(np.power(distance, 2)))


# In[127]:


def Ratio_SSD(feature_1, feature_2, feature_3):
    return SSD(feature_1, feature_2)/SSD(feature_1, feature_3)


# In[128]:


def Find_Matches(features_1: list, features_2: list, method: str = 'ssd',
                 threshold_ratio: float = 0.7, threshold: float = 0.2):
    
    distance_matrix = np.zeros((len(features_1), len(features_2)),dtype=np.float32)

    for row, feature_1 in enumerate(features_1):
        for col, feature_2 in enumerate(features_2):
            distance_matrix[row, col] = SSD(feature_1[2],feature_2[2])
    print(distance_matrix)
    matched_features = []
    for row, feature_1 in enumerate(features_1):
        best_match = np.argmin(distance_matrix[row])
#         print(best_match)
        if method == 'ssd':
            if distance_matrix[row, best_match] < threshold:
                matched_features.append([row, best_match, distance_matrix[row, best_match]])
        elif method == 'ratio_ssd':
            min_matches = np.argpartition(distance_matrix[row], (0, 1))[:2]
#             matched_features.append([row, min_matches, distance_matrix[row, min_matches[0]] / distance_matrix[row, min_matches[1]]])
            if (distance_matrix[row, min_matches[0]] / distance_matrix[row, min_matches[1]]) < threshold_ratio:
                matched_features.append([row, min_matches[0], distance_matrix[row, min_matches[0]]])
    return matched_features


# In[129]:


matched = Find_Matches(patches, patches_second, 'ratio_ssd')
# Test If Detected
# print(matched[0][0])
for i in range(len(matched)): 
    x1, y1 = patches[matched[i][0]][0], patches[matched[i][0]][1]
#     print("x1, y1: ",x1,y1)
    circles_1 = cv2.circle(real_img, (y1, x1), 3, (255, 0, 0), -1)
    cv2.imwrite("image_1.jpg", circles_1) 
    
for i in range(len(matched)):
    x2, y2 = patches_second[matched[i][1]][0], patches_second[matched[i][1]][1]
#     print("x2, y2: ",x2,y2)
    circles_2 = cv2.circle(real_img_second, (y2, x2), 3, (255, 0, 0), -1)
    cv2.imwrite("image_2.jpg", circles_2) 

