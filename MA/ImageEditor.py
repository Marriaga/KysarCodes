from builtins import object
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw

'''
This module loads an image and allows the user to paint the bone areas
and the membrane areas. The result is saved when the user presses "Show".

The result in an image that is the same resolution as the original but with
the markings of the user only.

The original intent of this was to specify what was bone and what was membrane
in an image of a bulge test such that the 3D data of the bone could be used to
fit the bone height between different pressure values and then the membrane data
could be used to build the 3D of the membrane as it is bulged.

The program works but I never really used it in the end.
'''



# Centered Circle Functions
def _create_circle(self, x, y, r, width=0, **kwargs):
    return self.create_oval(x-r+1, y-r+1, x+r, y+r, width=width,**kwargs)
tk.Canvas.create_circle = _create_circle

def _create_circle_PIL(self, x, y, r, fill="#000000"):
    return self.ellipse((x-r+1, y-r+1, x+r, y+r), outline=None, fill=fill)
ImageDraw.ImageDraw.create_circle =  _create_circle_PIL


WHITE = (255, 255, 255)

class ImgEditor(object):
    
    def __init__(self,FileImg):
        print("*Running Image Editor")
        self.root = tk.Tk()        
        
        #Picture
        self.FileImg = FileImg
        self.Img = Image.open(FileImg)
        self.ImRes = self.Img.size
        self.Scale = min(min(800/self.ImRes[0],600/self.ImRes[1]),1.0)
        self.NewImRes=(int(self.ImRes[0]*self.Scale),int(self.ImRes[1]*self.Scale))
        self.Imgtk = ImageTk.PhotoImage(self.Img.resize(self.NewImRes,Image.LANCZOS))
           
        self.BoneMembImg = Image.new("RGB", self.NewImRes, WHITE)
        self.draw = ImageDraw.Draw(self.BoneMembImg)
        
        #Canvas
        self.Cv = tk.Canvas(self.root, width=800 , height=600)
        self.Cobjects = []
        self.Instructions = []
        
        #Buttons
        self.BoneBtn = tk.Button(self.root, text='Bone', command=self.SetBone)
        self.MembBtn = tk.Button(self.root, text='Membrane', command=self.SetMembrane)
        self.SizeSld = tk.Scale(self.root, from_=1, to=40, orient=tk.HORIZONTAL, command=self.ChangeSizeSld)
        self.UndoBtn = tk.Button(self.root, text='Undo (Right-click)', command=self.Undo)
        self.ShowBtn = tk.Button(self.root, text='Show', command=self.Show)
        
        #Alignments
        self.BoneBtn.grid(row=0, column=0)
        self.MembBtn.grid(row=0, column=1)
        self.SizeSld.grid(row=0, column=2)
        self.UndoBtn.grid(row=0, column=3)
        self.ShowBtn.grid(row=0, column=4)
        self.Cv.grid(row=1, columnspan=6)
        
        #Set-up
        self.drawing=False
        self.undoing=False
        self.SizeSld.set(20)
        self.line_width = self.SizeSld.get()
        self.ActiveBtn = self.BoneBtn
        self.ActiveBtn.config(relief=tk.SUNKEN)
        self.Color = "#ff0000"
        self.Cv.bind("<Motion>", self.Motion)
        self.Cv.bind("<ButtonPress-1>", self.B1Down)
        self.Cv.bind("<ButtonRelease-1>", self.B1Up)
        self.Cv.bind("<ButtonPress-3>", self.B3Down)
        self.Cv.bind("<ButtonRelease-3>", self.B3Up)
        
        self.ioc = self.Cv.create_image(0, 0, anchor="nw", image=self.Imgtk)
        
        self.root.mainloop()
        
    def ChangeSizeSld(self,event):
        self.line_width = self.SizeSld.get()
        
    def Show(self):
        self.BoneMembImg = Image.new("RGB", self.NewImRes, WHITE)
        self.draw = ImageDraw.Draw(self.BoneMembImg)
        for x,y,r,c in self.Instructions:
            self.draw.create_circle(x,y,r,fill=c)
        Timg=self.BoneMembImg.resize(self.ImRes,Image.NEAREST)
        Timg.save(self.FileImg[:-4] + '_BM.png')
        Timg.show()
        
    def SetBone(self):
        self.ActivateBtn(self.BoneBtn)
        self.Color="#ff0000"
        
    def SetMembrane(self):
        self.ActivateBtn(self.MembBtn)
        self.Color="#0000ff"
        
    def ActivateBtn(self, NewBtn):
        self.ActiveBtn.config(relief=tk.RAISED)
        self.ActiveBtn = NewBtn
        self.ActiveBtn.config(relief=tk.SUNKEN)
    
    def B1Down(self,event):
        self.drawing=True
        self.DrawCircle(event.x, event.y, self.line_width/2)
        
    def B1Up(self,event):
        self.drawing=False
    
    def B3Down(self,event):
        self.undoing=True
        self.Undo()
        
    def B3Up(self,event):
        self.undoing=False
        
    def Motion(self,event):
        if self.drawing:
            self.DrawCircle(event.x, event.y, self.line_width/2)
        elif self.undoing:
            self.Undo()
    
    def DrawCircle(self,x,y,r):
        self.Cobjects.append(self.Cv.create_circle(x,y,r, fill=self.Color))
        self.Instructions.append((x,y,r,self.Color)) 

    def Undo(self):
        if len(self.Cobjects)>0:
            self.Cv.delete(self.Cobjects.pop())
            self.Instructions.pop()

if __name__ == "__main__":

    ge=ImgEditor("ImageEditorExample.png")