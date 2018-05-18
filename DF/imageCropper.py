#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 21:14:26 2018

module to crop images iteratively and save them in a directory 

@author: df
"""

import os

from PIL import Image

def main():
    """
    """
    im_dir = '/Users/df/Documents/0-DATA/s1_20180223/MAX_20180223_S1_20X_3c/'
    im_name = 'MAX_20180223_S1_20X_3c.png'
    # im_name_0 = os.path.splitext(im_name)[0]
    im_path = im_dir + im_name
    
    num = 5
    
    cropCenter( im_path, num )
    
    cropUpperLeft( im_path, num )
    
    cropUpperRightRect( im_path, num )
    
    cropUpperRightSq( im_path, num )
    
    cropLowerLeftRect( im_path, num )
    
    cropLowerLeftSq( im_path, num )
    
    cropLowerRight( im_path, num )
    

def cropCenter( image_path, num ):
    """
    @param num: number of cropped images to produce
    @param image_path: The path to the image to edit
    """
    temp_path, im_name = os.path.split(image_path)
    im_name_0 = os.path.splitext(im_name)[0]
    save_path = temp_path + '/CenterCropped/'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    width, height, cen_x, cen_y = imageDimensions( image_path )
    
    rx = int(width/num)
    ry = int(height/num)
    
    for ii in range(1,num):
        x0 = cen_x - ii*rx/2
        y0 = cen_y - ii*ry/2
        xn = x0 + ii*rx
        yn = y0 + ii*ry
        coords = ( x0, y0, xn, yn )
        # print(coords)
        saved_location = save_path + im_name_0 + '_' + 'CenterCropped' \
                                                            + str(ii) + '.png'
        cropp( image_path, coords, saved_location )
   
    
def cropUpperLeft( image_path, num ):
    """
    @param num: number of cropped images to produce
    @param image_path: The path to the image to edit
    """
    temp_path, im_name = os.path.split(image_path)
    im_name_0 = os.path.splitext(im_name)[0]
    save_path = temp_path + '/UpperLeftCropped/'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    width, height, cen_x, cen_y = imageDimensions( image_path )
    
    rx = int(width/num)
    ry = int(height/num)
    
    x0 = 0
    y0 = 0
    for ii in range(1,num):
        xn = ii*rx
        yn = ii*ry
        coords = ( x0, y0, xn, yn )
        saved_location = save_path + im_name_0 + '_' + 'UpperLeftCropped' \
                                                            + str(ii) + '.png'
        cropp( image_path, coords, saved_location )
    
    
def cropUpperRightRect( image_path, num ):
    """
    @param num: number of cropped images to produce
    @param image_path: The path to the image to edit
    """
    temp_path, im_name = os.path.split(image_path)
    im_name_0 = os.path.splitext(im_name)[0]
    save_path = temp_path + '/UpperRightRectCropped/'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    width, height, cen_x, cen_y = imageDimensions( image_path )
    
    rx = int(width/num)
    ry = int(height/num)
    
    xn = width
    y0 = 0
    for ii in range(1,num):
        x0 = width - ii*rx
        yn = height - ii*ry
        coords = ( x0, y0, xn, yn )
        saved_location = save_path + im_name_0 + '_' + 'UpperRightRectCropped'\
                                                            + str(ii) + '.png'
        cropp( image_path, coords, saved_location )
        

def cropUpperRightSq( image_path, num ):
    """
    @param num: number of cropped images to produce
    @param image_path: The path to the image to edit
    """
    temp_path, im_name = os.path.split(image_path)
    im_name_0 = os.path.splitext(im_name)[0]
    save_path = temp_path + '/UpperRightSqCropped/'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    width, height, cen_x, cen_y = imageDimensions( image_path )
    
    rx = int(width/num)
    ry = int(height/num)
    
    xn = width
    y0 = 0
    for ii in range(1,num):
        x0 = width - ii*rx
        yn = ii*ry
        coords = ( x0, y0, xn, yn )
        saved_location = save_path + im_name_0 + '_' + 'UpperRightSqCropped' \
                                                            + str(ii) + '.png'
        cropp( image_path, coords, saved_location )
    

def cropLowerLeftRect( image_path, num ):
    """
    @param num: number of cropped images to produce
    @param image_path: The path to the image to edit
    """
    temp_path, im_name = os.path.split(image_path)
    im_name_0 = os.path.splitext(im_name)[0]
    save_path = temp_path + '/LowerLeftRectCropped/'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    width, height, cen_x, cen_y = imageDimensions( image_path )
    
    rx = int(width/num)
    ry = int(height/num)
    
    x0 = 0
    yn = height
    for ii in range(1,num):
        xn = ii*rx
        y0 = ii*ry
        coords = ( x0, y0, xn, yn )
        saved_location = save_path + im_name_0 + '_' + 'LowerLeftRectCropped' \
                                                            + str(ii) + '.png'
        cropp( image_path, coords, saved_location )


def cropLowerLeftSq( image_path, num ):
    """
    @param num: number of cropped images to produce
    @param image_path: The path to the image to edit
    """
    temp_path, im_name = os.path.split(image_path)
    im_name_0 = os.path.splitext(im_name)[0]
    save_path = temp_path + '/LowerLeftSqCropped/'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    width, height, cen_x, cen_y = imageDimensions( image_path )
    
    rx = int(width/num)
    ry = int(height/num)
    
    x0 = 0
    yn = height
    for ii in range(1,num):
        xn = ii*rx
        y0 = height - ii*ry
        coords = ( x0, y0, xn, yn )
        saved_location = save_path + im_name_0 + '_' + 'LowerLeftSqCropped' \
                                                            + str(ii) + '.png'
        cropp( image_path, coords, saved_location )


def cropLowerRight( image_path, num ):
    """
    @param num: number of cropped images to produce
    @param image_path: The path to the image to edit
    """
    temp_path, im_name = os.path.split(image_path)
    im_name_0 = os.path.splitext(im_name)[0]
    save_path = temp_path + '/LowerRightCropped/'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    width, height, cen_x, cen_y = imageDimensions( image_path )
    
    rx = int(width/num)
    ry = int(height/num)
    
    xn = width
    yn = height
    for ii in range(1,num):
        x0 = width - ii*rx
        y0 = height - ii*ry
        coords = ( x0, y0, xn, yn )
        saved_location = save_path + im_name_0 + '_' + 'LowerRightCropped' \
                                                            + str(ii) + '.png'
        cropp( image_path, coords, saved_location )


def cropp( image_path, coords, saved_location ):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    print(str(saved_location))
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    cropped_image.show()
    

def imageDimensions( image ):
    """
    function to store into an array the dimensions and the center of an image
    """
    # width and height of image object in pixels:
    image_obj = Image.open(image)
    width, height = image_obj.size
    # center coordinates of image in pixels:
    cen_x = int(width/2)
    cen_y = int(height/2)
    imDim = ( width, height, cen_x, cen_y )
    
    return imDim
