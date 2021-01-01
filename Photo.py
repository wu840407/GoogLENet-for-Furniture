#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
filename=r'./img'
def Photo(filename):
    for file in os.listdir(filename):
        img=cv2.imread(filename+'/'+file,cv2.IMREAD_GRAYSCALE)
        f=open('./anno/'+file.split(".")[0]+'.txt')
        #print(file)
        oh,ow=img.shape[:2]
        count=0
        for line in f.readlines():
            #print(line)
            w,h=round(float(line.split(" ")[3])*ow),round(float(line.split(" ")[4])*oh)
            x,y=round(float(line.split(" ")[1])*ow-w/2),round(float(line.split(" ")[2])*oh-h/2)
            crop_img=img[y:y+h,x:x+w]
            cv2.resize(crop_img,(224,224))
            cv2.imwrite('./train/'+file.split(" ")[0]+'_'+str(count)+'_'+line.split(" ")[0]+'.jpg',crop_img)
        f.close()
if __name__ == '__main__':
    Photo(filename)

