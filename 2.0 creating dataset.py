import os
# for image to csv
import cv2
import dlib
#search about dlib library
from imutils import face_utils
import numpy as np

datapath='Face shapes'#this path is loacted in my code location

labels=os.listdir(datapath)
#to scan inside files and create the list
#os.mkdir('MyFolder/text.txt')#crete new folder

#print(files)
label_dict={}

for i in range(len(labels)):

    label_dict[labels[i]]=i#update dictionary with newly type labels everytime
print(label_dict)


face_detector=dlib.get_frontal_face_detector()
#loading the pretrained algorithm for face detection available in dlib

landmark_detector=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#loading the pretrained algorith for 68 landmark detecting externelly

data=[]
target=[]

def measure_distances(img,label):

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #COLOR_BGR2FRAY - color conversion flag

    rects=face_detector(gray)
    #same like- algorithm.predict(test_data)

    for rect in rects :

        x1=rect.left()
        y1=rect.top()
        x2=rect.right()
        y2=rect.bottom()

        

        points=landmark_detector(gray, rect)
        #passing the gray img and ROI(detected face rectangle) to the landmark_detector
        #return is 68 points (x,y)(NOTE: this is not a numpy array)
        #we have to convetr into numpy array

        points=face_utils.shape_to_np(points)

        #to get our necessary points
        myPoints=points[2:9,0]#0 means x coordinate
        #grtting the x cordinates of point 2-8

        D1=myPoints[6]-myPoints[0]
        D2=myPoints[6]-myPoints[1]
        D3=myPoints[6]-myPoints[2]
        D4=myPoints[6]-myPoints[3]
        D5=myPoints[6]-myPoints[4]
        D6=myPoints[6]-myPoints[5]#getting distances from one point to another point

        d1=D2/D1
        d2=D3/D1
        d3=D4/D1
        d4=D5/D1
        d5=D6/D1

        #store in a array
        data.append([d1,d2,d3,d4,d5])
        #if(label=='Diamond')
           #target.append()--this is primary method
        #we have to implement dictionary
        target.append(label)
        
    
for label in labels:

    imgs_path=os.path.join(datapath,label)
    
    #print(imgs_path)
    img_names=os.listdir(imgs_path)
    #print(img_names)#names of the images

    for img_name in img_names :
        
        img_path=os.path.join(imgs_path,img_name)
        img=cv2.imread(img_path)

        cv2.imshow('LIVE',img)
        cv2.waitKey(100)#load one by one images

        measure_distances(img,label_dict[label])

        
        
print(data)
print(target)

#to save data and target as a external file
#save data and target in different file to maximuze the efficieny of the program.
np.save('data', data)
np.save('target',target)
