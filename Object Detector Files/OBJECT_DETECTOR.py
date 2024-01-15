#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install opencv-python


# In[2]:


import cv2 #cv2 is an open source computer vision/image


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


#empty list for the class labels
classLabels = []

#file path for the labels.txt file
file_name = r'C:\Users\jujun\OneDrive\Desktop\Object Detector Files\Labels.txt'

#this code will read the class labels from the labels.txt file
with open(file_name, 'r') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')


# In[5]:


#file path of the configuration file and the frozen model file
config_file = r'C:\Users\jujun\OneDrive\Desktop\Object Detector\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = r'C:\Users\jujun\OneDrive\Desktop\Object Detector\frozen_inference_graph.pb'

#create model using the configuration file and the frozen model file above
model = cv2.dnn_DetectionModel(frozen_model, config_file)


# In[6]:


#this will print the texts or the labels inside the labels.txt file
print(classLabels)


# In[7]:


#this will print the length of the file
print(len(classLabels))


# In[8]:


model.setInputSize(320, 320) #size
model.setInputScale(1.0 / 127.5) #scale
model.setInputMean((127.5, 127.5, 127.5)) #mean
model.setInputSwapRB(True) #swapr&b


# In[9]:


#This code will read the image of the given file 

img = cv2.imread(r'C:\Users\jujun\OneDrive\Desktop\Object Detector Files\R.png')
if img is not None:
    plt.imshow(img)
    plt.show()
else:
    print("Error loading the image.")


# In[10]:


#this will display the image using Matplotlib after converting from BGR to RGB color space
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[11]:


#this detect objects in the input image using the trained model.
ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)


# In[12]:


print(ClassIndex)


# In[13]:


for ClassInd, conf, box in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    print(f"Class Index: {ClassInd}, Confidence: {conf}, Bounding Box: {box}")
    


# In[14]:


# Assuming ClassIndex, confidence, and bbox are obtained from the detection
for ClassInd, conf, box in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    x, y, w, h = box
    color = (255, 0, 0)  # BGR color format, so (255, 0, 0) is blue

    #this will rectangle
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

    #this will put text
    label = f"Class: {ClassInd}, Confidence: {conf:.2f}"
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# In[15]:


font_scale = 3 #scale
font = cv2.FONT_HERSHEY_PLAIN #font

for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)
    cv2.rectangle(img, boxes, (255, 0, 0), 2)
    cv2.putText(img, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)


# In[16]:


#this will print the final result/the detected image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[46]:


#THE SECOND IMAGE


# In[17]:


#This code will read the image of the given file 

img = cv2.imread(r'C:\Users\jujun\OneDrive\Desktop\Object Detector Files\yellow_cayman_gt4_decals-600x400.jpg')
if img is not None:
    plt.imshow(img)
    plt.show()
else:
    print("Error loading the image.")


# In[18]:


#this will display the image using Matplotlib after converting from BGR to RGB color space
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[19]:


ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)
for ClassInd, conf, box in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    print(f"Class Index: {ClassInd}, Confidence: {conf}, Bounding Box: {box}")


# In[20]:


# Assuming ClassIndex, confidence, and bbox are obtained from the detection
for ClassInd, conf, box in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    x, y, w, h = box
    color = (255, 0, 0)  # BGR color format, so (255, 0, 0) is blue

    #this will draw rectangle
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

    #this will put text
    label = f"Class: {ClassInd}, Confidence: {conf:.2f}"
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)
    cv2.rectangle(img, boxes, (255, 0, 0), 2)
    cv2.putText(img, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)


# In[21]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[ ]:


#THE THIRD IMAGE


# In[22]:


#this will read the image

img = cv2.imread(r'C:\Users\jujun\OneDrive\Desktop\Object Detector Files\OIP.jpg')
if img is not None:
    plt.imshow(img)
    plt.show()
else:
    print("Error loading the image.")


# In[23]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[24]:


ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)


# In[25]:


print(ClassIndex)


# In[26]:


for ClassInd, conf, box in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    print(f"Class Index: {ClassInd}, Confidence: {conf}, Bounding Box: {box}")
    


# In[27]:


# Assuming ClassIndex, confidence, and bbox are obtained from the detection
for ClassInd, conf, box in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    x, y, w, h = box
    color = (255, 0, 0)  # BGR color format, so (255, 0, 0) is blue

    # Draw rectangle
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

    # Put text
    label = f"Class: {ClassInd}, Confidence: {conf:.2f}"
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)
    cv2.rectangle(img, boxes, (255, 0, 0), 2)
    cv2.putText(img, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)    


# In[28]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[ ]:


#THE FOURTH IMAGE


# In[29]:


# read an image

img = cv2.imread(r'C:\Users\jujun\OneDrive\Desktop\Object Detector Files\F122_9773.jpg')
if img is not None:
    plt.imshow(img)
    plt.show()
else:
    print("Error loading the image.")


# In[30]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[31]:


ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)


# In[32]:


print(ClassIndex)


# In[33]:


for ClassInd, conf, box in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    print(f"Class Index: {ClassInd}, Confidence: {conf}, Bounding Box: {box}")
    


# In[34]:


# Assuming ClassIndex, confidence, and bbox are obtained from the detection
for ClassInd, conf, box in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    x, y, w, h = box
    color = (255, 0, 0)  # BGR color format, so (255, 0, 0) is blue

    # Draw rectangle
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

    # Put text
    label = f"Class: {ClassInd}, Confidence: {conf:.2f}"
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# In[35]:


font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)
    cv2.rectangle(img, boxes, (255, 0, 0), 2)
    cv2.putText(img, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)


# In[36]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[ ]:


#THE FIFTH IMAGE


# In[37]:


# read an image

img = cv2.imread(r'C:\Users\jujun\OneDrive\Desktop\Object Detector Files\rsz_young_peoples_needs.jpg')
if img is not None:
    plt.imshow(img)
    plt.show()
else:
    print("Error loading the image.")


# In[38]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[39]:


ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)


# In[40]:


print(ClassIndex)


# In[41]:


for ClassInd, conf, box in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    print(f"Class Index: {ClassInd}, Confidence: {conf}, Bounding Box: {box}")
    


# In[42]:


# Assuming ClassIndex, confidence, and bbox are obtained from the detection
for ClassInd, conf, box in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    x, y, w, h = box
    color = (255, 0, 0)  # BGR color format, so (255, 0, 0) is blue

    # Draw rectangle
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

    # Put text
    label = f"Class: {ClassInd}, Confidence: {conf:.2f}"
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# In[43]:


font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)
    cv2.rectangle(img, boxes, (255, 0, 0), 2)
    cv2.putText(img, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)


# In[44]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[ ]:




