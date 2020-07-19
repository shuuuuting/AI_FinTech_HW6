#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np 
from PIL import Image
from scipy import signal
 
I = Image.open('IMG_6139.JPG')
data = np.asarray(I)
#(1)
W,H = I.size
noise = np.random.normal(1,25,(H,W,3))
data1 = data+noise
data1[data1>255] = 255
data1[data1<0] = 0
data1 = data1.astype('uint8')
I1 = Image.fromarray(data1,'RGB')
I1.show()

#(2)
x,y = np.meshgrid(np.linspace(-1,1,8),np.linspace(-1,1,8))
d = np.sqrt(x*x+y*y)
sigma,mu = 0.5,0.0
mask = np.exp(-((d-mu)**2/(2.0*sigma**2))) #高斯模糊
mask = mask/np.sum(mask[:])
R = data1[:,:,0]
G = data1[:,:,1]
B = data1[:,:,2]
R2 = signal.convolve2d(R,mask,boundary='symm',mode='same') #對稱方法,輸出跟輸入一樣大小
G2 = signal.convolve2d(G,mask,boundary='symm',mode='same')
B2 = signal.convolve2d(B,mask,boundary='symm',mode='same')
data2 = data1.copy()
data2[:,:,0] = R2.astype('uint8')
data2[:,:,1] = G2.astype('uint8')
data2[:,:,2] = B2.astype('uint8')
I2 = Image.fromarray(data2,'RGB')
I2.show()

#(3)
x,y = np.meshgrid(np.linspace(-1,1,15),np.linspace(-1,1,15))
d = np.sqrt(x*x+y*y)
sigma,mu = 15,0.0
mask = np.exp(-((d-mu)**2/(2.0*sigma**2))) #高斯模糊
mask = mask/np.sum(mask[:])
R = data[:,:,0]
G = data[:,:,1]
B = data[:,:,2]
R3 = signal.convolve2d(R,mask,boundary='symm',mode='same') #對稱方法,輸出跟輸入一樣大小
G3 = signal.convolve2d(G,mask,boundary='symm',mode='same')
B3 = signal.convolve2d(B,mask,boundary='symm',mode='same')
data3 = data.copy()
data3[:,:,0] = R3.astype('uint8')
data3[:,:,1] = G3.astype('uint8')
data3[:,:,2] = B3.astype('uint8')
I3 = Image.fromarray(data3,'RGB')
I3.show()

#(4)
data = np.asarray(I.convert('L'))
mx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype=np.float)
my = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype = np.float)
Ix = signal.convolve2d(data,mx,boundary='symm',mode='same')
Iy = signal.convolve2d(data,my,boundary='symm',mode='same')
sketch = np.square(Ix) + np.square(Iy)
sketch[sketch<sketch.max()*10/100] = 255
sketch[sketch>=sketch.max()*10/100] = 0
I4 = Image.fromarray(sketch)
I4.show()
