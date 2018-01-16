from __future__ import division
import numpy as np
import os, glob
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
import MAPyLibs.Tools as MATL
import matplotlib.pyplot as plt

def PlotAngles(Name,A,Is,cols):
    if not type(Is[0])==type([]): Is=[Is]
    fig = plt.figure(figsize=(8,6))
    axs  = fig.add_subplot(111)
    for i,Iy in enumerate(Is):
        axs.plot(A,Iy,marker='.',color=cols[i],ls='-',markersize=5,label='P'+str(i+1))

    axs.set_ylabel("Intensity")
    axs.set_xlabel("Angle")
    #axs.set_xbound([-3,3])
    axs.grid(True,linewidth=0.5)
    axs.set_axisbelow(True)
    
    hand, labl = axs.get_legend_handles_labels()
    axs.legend(hand,labl, bbox_to_anchor=(0.95,0.01), shadow=True, fancybox=True,loc='lower right',ncol=1,handlelength=2.5,labelspacing=0.3,columnspacing=0.8,prop={'size':12})
    
    fig.savefig(Name+'.png')
    plt.close(fig)    

def MakeImgCirc(OutName,ImgName,pos,cols):
        if not type(pos[0]) == type(()): pos = [pos]
        r=6
        BImg = Image.open(ImgName)
        Draw = ImageDraw.Draw(BImg)
        font = ImageFont.truetype("ariblk.ttf", 9)
        i=1
        for x,y in pos:
            Draw.ellipse((x-r+1, y-r+1, x+r, y+r), outline='black', fill=cols[i-1])
            Draw.text((x+7, y+7),str(i),cols[i-1],font=font)
            i+=1

        BImg.save(OutName+".png")

def GetIntesity(Name,pos):
    Img = Image.open(Name).convert('L')
    return Img.getpixel(pos)
    

def GetXMLAngle(file):
    tree = ET.parse(file)
    root = tree.getroot()
    RigTrsf = root.findall("*[@class='mpicbg.trakem2.transform.RigidModel2D']")
    if RigTrsf:
        return float(RigTrsf[0].get('data').split()[0])*180/np.pi
    else:
        return 0.0

    

    
    
    
    
#Registered Virtual Stack Images with Transformation Output Folder
SRC="C:\\Users\\Miguel\\Work\\RWM-Polarized\\2017-08-24 Polarized\\ARegLowMag"
NAME="RWM_10X"
Bname=os.path.join(SRC,NAME)

Ltif = MATL.getFiles(SRC,"tif",NAME)
Lxml = MATL.getFiles(SRC,"xml",NAME)

posL=  [(1083,1065),(970,1127),(1053,1007),(995,1044),(988,1065)]
cols = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00'] #http://colorbrewer2.org

A=[]
I=[]
for tif,xml in zip(Ltif,Lxml):
    A.append(GetXMLAngle(xml))
    I.append([])
    for pos in posL:
        I[-1].append(GetIntesity(tif,pos))
        
A.append(-180)
I.append(I[0])        
              
I=[list(i) for i in zip(*I)]

PlotAngles(Bname+"_Angles",A,I,cols)
MakeImgCirc(Bname+"_Points",Ltif[0],posL,cols)   






 
