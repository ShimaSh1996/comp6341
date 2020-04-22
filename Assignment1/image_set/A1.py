#!/usr/bin/env python
# coding: utf-8

# # Computer Vision 

# ### Assignment 1

# Please run the cell for each Bayer image and its corresponding one at a time. This notebook should be in the same folder as test images. Also the results will be saved in the same folder. 

# In[97]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[98]:


import cv2
import numpy as np
from matplotlib import pyplot as plt


# First Image

# In[99]:


#1
path = 'pencils_mosaic.bmp'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)/255


# In[100]:


#1
path = 'pencils.jpg'
real_img = cv2.imread(path).astype(np.float32)/255


# Second Image

# In[101]:


#2
path = 'crayons_mosaic.bmp'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)/255


# In[102]:


#2
path = 'crayons.jpg'
real_img = cv2.imread(path).astype(np.float32)/255


# Third Image

# In[103]:


#3
path = 'oldwell_mosaic.bmp'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)/255


# In[104]:


#3
path = 'oldwell.jpg'
real_img = cv2.imread(path).astype(np.float32)/255


# ### Part 1)

# In[105]:


red = np.zeros(img.shape)
blue = np.zeros(img.shape)
green = np.zeros(img.shape)


# Seperate color channels

# In[106]:


for i in range (img.shape[1]):
    for j in range (img.shape[0]):
        if (i+j)%2 == 1:
            red[j,i] = img[j,i]
        elif j%2 == 0:
            blue[j,i] = img[j,i]
        else:
            green[j,i] = img[j,i]


# Kernel Definition

# In[107]:


kernel_red = np.array([[0, 0.25, 0], [0.25, 1, 0.25], [0, 0.25, 0]])
kernel_bag = np.array([[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]])


# In[108]:


new_blue = cv2.filter2D(blue, -1, kernel_bag).astype(np.float32)
new_red = cv2.filter2D(red, -1, kernel_red).astype(np.float32)
new_green = cv2.filter2D(green, -1, kernel_bag).astype(np.float32)


# Image Reconstruction

# In[109]:


new_img = cv2.merge(((new_blue), (new_green), (new_red)))


# Calculating the difference

# In[110]:


channelwise_dif = np.sum((real_img - new_img)**2, axis = -1)
# channelwise_dif_sqrt = np.sqrt(np.square(real_img[:,:,2] - new_red) +
#                           np.square(real_img[:,:,1] - new_green) + 
#                           np.square(real_img[:,:,0] - new_blue) )
# channelwise_dif_square = np.sum(np.square((real_img - new_img)**2), axis = -1)
# cv2.imshow("test", channelwise_dif_sqrt)
# cv2.waitKey(100)


# Saving all together

# In[111]:


dif = np.zeros(real_img.shape)
print(channelwise_dif.shape)
dif[:,:,0] = channelwise_dif*255

final_res_dif = np.concatenate([real_img*255, new_img*255, dif], axis = 1) 


# Save the results

# In[112]:


cv2.imwrite('new_img.png', new_img*255)
cv2.imwrite('dif_img.png', final_res_dif)
cv2.imwrite('channlewise_dif.png', channelwise_dif*255)


# In[66]:


#cv2.imshow("dif_img", channelwise_dif*255)
#cv2.imshow("new_img", new_img*255)
#cv2.imshow("final_res_dif", final_res_dif)
#cv2.waitKey(100)


# ### Part 2)

# In[67]:


green_red = new_green - new_red
blue_red = new_blue - new_red

median_green = cv2.medianBlur(green_red, 3)
median_blue = cv2.medianBlur(blue_red, 3)

second_green = (median_green + new_red)
second_blue = (median_blue + new_red)


# In[68]:


second_img = cv2.merge(((second_blue), (second_green), (new_red)))


# In[69]:


channelwise_dif_2 = np.sum((real_img - second_img)**2, axis = -1)

# Uncomment the following lines of code to see the result
# channelwise_dif_2 = np.sqrt(np.square(real_img[:,:,2] - second_img[:,:,2]) +
#                           np.square(real_img[:,:,0] - second_img[:,:,0]) + 
#                           np.square(real_img[:,:,1] - second_img[:,:,1])).astype(np.uint8)

# cv2.imshow("channelwise_dif_2", channelwise_dif_2)
# cv2.waitKey(100)
# cv2.imshow("second_dif", second_dif)
# cv2.waitKey(100)


# In[70]:


dif2 = np.zeros(real_img.shape)
print(channelwise_dif.shape)
dif2[:,:,0] = channelwise_dif*255

second_final_res_dif = np.concatenate([real_img*255, second_img*255, dif2],axis = 1) 


# In[71]:


cv2.imwrite('second_img.png', second_img*255)
cv2.imwrite('channelwise_dif_2.png', channelwise_dif_2*255)
cv2.imwrite('second_final_res_dif.png', second_final_res_dif)


# ## Some Experiment

# In[72]:


bayer_img = np.zeros(img.shape)


# In[73]:


for i in range (real_img.shape[1]):
    for j in range (real_img.shape[0]):
        if (i+j)%2 == 1:
            bayer_img[j,i] = real_img[j,i,1]
            #print(new_bayer[j,i])
        elif j%2 == 0:
            bayer_img[j,i] = real_img[j,i,0]
        else:
            bayer_img[j,i] = real_img[j,i,2]


# In[74]:


#bayer_img


# In[75]:


red_2 = np.zeros(bayer_img.shape)
blue_2 = np.zeros(bayer_img.shape)
green_2 = np.zeros(bayer_img.shape)


# In[76]:


for i in range (bayer_img.shape[1]):
    for j in range (bayer_img.shape[0]):
        if (i+j)%2 == 1:
            green_2[j,i] = bayer_img[j,i]
        elif j%2 == 0:
            blue_2[j,i] = bayer_img[j,i]
        else:
            red_2[j,i] = bayer_img[j,i]


# In[77]:


kernel_green = np.array([[0, 0.25, 0], [0.25, 1, 0.25], [0, 0.25, 0]])
kernel_bar = np.array([[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]])


# In[78]:


new_blue_2 = cv2.filter2D(blue_2, -1, kernel_bar)
new_red_2 = cv2.filter2D(red_2, -1, kernel_bar)
new_green_2 = cv2.filter2D(green_2, -1, kernel_green)


# In[79]:


new_img_2 = cv2.merge(((new_blue_2), (new_green_2), (new_red_2)))
#real_img = real_img.astype(np.float64)


# In[80]:


channelwise_dif_3 = np.sum((real_img - new_img_2)**2, axis = -1)


# In[81]:


dif3 = np.zeros(real_img.shape)
dif3[:,:,0] = channelwise_dif_3*255

second_final_res_dif = np.concatenate([real_img*255, new_img_2*255, dif3],axis = 1) 

final_res_dif3 = np.concatenate([real_img*255, new_img_2*255, dif3],axis = 1) 


# In[82]:


cv2.imwrite('new_img3.png', new_img_2*255)
cv2.imwrite('channelwise_dif_3.png', channelwise_dif_3*255)
cv2.imwrite('final_res_dif3.png', final_res_dif3)


# In[ ]:





# In[ ]:





# In[ ]:




