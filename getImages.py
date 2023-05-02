import os
import numpy as np
import sys
from PIL import ImageTk, Image, ImageOps
from tempfile import TemporaryFile

testImages = TemporaryFile()

allPixels=[]
i=0
if __name__ == '__main__':
  dir_path = os.path.dirname(os.path.realpath(__file__))
  # f = open('bruh.csv','w')
  for root, dirs, files in os.walk(dir_path):
    for dir in dirs:
        for file_name in os.listdir(dir):
            if file_name.endswith('.jpg'):
                # f.write(dir)
                # f.write('/')
                # f.write(file_name[:-10])
                # f.write('\n')
                img=Image.open(dir+'/'+file_name)
                img=img.resize((round(img.size[0]*0.2), round(img.size[1]*0.2)))
                img=ImageOps.grayscale(img)
                #np_image=np.array(img).flatten()
                np_image=np.array(img)
                #allPixels=np.vstack([allPixels,img])
                allPixels.append(np_image)
                #np.append(allPixels,np_image)
                if(i%100==0):
                   print(i)
                i=i+1
npPixels=np.array(allPixels)           
np.save('testImages.npy',npPixels) 

#np.savetxt('testImagesSmall.txt', flattened, fmt='%d')
#np.savetxt("ree.csv", allPixels, delimiter=",")