from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import zip
from builtins import str
from builtins import range
import numpy as np
import os, glob
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
import MA.Tools as MATL
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

    
def ScanPoints(InputFolder,ImageExtension,ImagePrefix=None,PointsList=[(0,0)]):
    Limg = MATL.getFiles(InputFolder,ImageExtension,ImagePrefix)
    Lxml = MATL.getFiles(InputFolder,"xml",ImagePrefix)

    #Get Angles
    A=[]
    for xmlf in Lxml:
        A.append(GetXMLAngle(xmlf))
    A.append(-180)

    #Get Inensities
    I=[]
    for imgf in Limg:
        I.append([])
        for pos in PointsList:
            I[-1].append(GetIntesity(imgf,pos))
    I.append(I[0])           
    I=[list(i) for i in zip(*I)]
    return A,I,Limg[0]

def BuildPlots(InpupFolder,OutputFolder,ImageExtension,ImagePrefix=None,PointsList=[(0,0)]):
    """ Function that builds the Intensity vs. Angle of a sequence of registered images.

    Images must be pre-registered with ImageJ by doing Plugins>Registration>Register Virtual Stack Slices,
    making sure that the "Save Transforms" checkbox is selected and useing the same output folder for the 
    regiostered images and the transformation xml files
    
    Args:
        InpupFolder (str):
        The path to the input files, both the images and the xml files

        OutputFolder (str):
        The path to the output folder for the generated plots

        ImageExtension (str):
        The extension of the image files ("png","tiff", etc...)

        ImagePrefix (str):
        The prefix common to all images

        PointsList (list of tuples):
        Points for which to compute the intensity (default: [(0,0)])
    
    """

    A,I,refimg = ScanPoints(InpupFolder,ImageExtension,ImagePrefix=ImagePrefix,PointsList=PointsList)
    BaseName=os.path.join(OutputFolder,ImagePrefix)
    ColSequence = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999'] #http://colorbrewer2.org
    cols = [ColSequence[i] for i in range(len(PointsList))]
    PlotAngles(BaseName+"_Angles",A,I,cols)
    MakeImgCirc(BaseName+"_Points",refimg,PointsList,cols) 


if __name__ == '__main__':
    # Example of usage
    InpupFolder="C:\\Users\\Miguel\\Work\\RWM-Polarized\\2017-08-24 Polarized\\ARegLowMag"
    OutputFolder="C:\\Users\\Miguel\\Desktop\\NewOut"
    ImageExtension="tif"
    ImagePrefix="RWM_10X"
    PointsList=[(1083,1065),(970,1127),(1053,1007),(995,1044),(988,1065)]
    BuildPlots(InpupFolder,OutputFolder,ImageExtension,ImagePrefix,PointsList)
