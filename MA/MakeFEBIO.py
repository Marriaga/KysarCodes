import os
import matplotlib.pyplot as plt
import matplotlib.offsetbox as mplob
import xml.dom.minidom as minidom
import numpy as np
import pickle

import MA.MeshIO as MAIO
import MA.Tools as MATL
import MA.FigureProperties as MAFP

import pandas as pd

pjoin = os.path.join

def getdata(lines):
    # with open(logfile,"r") as fp:
    #     lines = fp.readlines()
    TIMES=[]
    DISPS=[]
    for i in range(len(lines)):
        if "Data = uz" in lines[i]:
            time=float(lines[i-1].strip().split()[-1])
            disp=float(lines[i+1].strip().split()[1])
            TIMES.append(time)
            DISPS.append(disp)
    return np.array(TIMES),np.array(DISPS)

def save_obj(savefile, theobject):
    with open(savefile, 'wb') as fp:
        pickle.dump(theobject, fp, pickle.HIGHEST_PROTOCOL)

def load_obj(savefile):
    with open(savefile, 'rb') as fp:
        return pickle.load(fp)

def fixlabel(l,Geo=False,Side=False,Fiber=False,Mesh=False):

    Geo_l=""
    if Geo:
        if "LP" in l:
            Geo_l = "Membrane;"
        else:
            Geo_l = "Disk;"
    
    Side_l=""
    if Side:
        if "IE" in l:
            Side_l = "Inner Ear Pressure;"
        else:
            Side_l = "Middle Ear Pressure;"

    Fiber_l=""
    if Fiber:
        if "Uniform" in l:
            Fiber_l = " Uniform Dispersion;"
        elif "Align" in l:
            Fiber_l = " Fully Aligned;"
        else:
            Fiber_l = " Avg. Dispersion;"

    Mesh_l=""
    if Mesh:
        if "2k" in l:
            Mesh_l = "Coarse Mesh;"
        elif "5k" in l:
            Mesh_l = "Medium Mesh;"
        elif "10k" in l:
            Mesh_l = "Fine Mesh;"

    label = " ".join(" ".join([Geo_l,Side_l,Fiber_l,Mesh_l]).split())

    return label

def getcml(lab,Color=["XX"],Marker=["XX"],Line=["XX"]):
    colors = ["xkcd:blue","xkcd:red","xkcd:green","xkcd:goldenrod","xkcd:turquoise","xkcd:royal purple"]
    markers = ["o","s","^",".","d"]
    lines = ["-","--",":"]

    c='k'
    for i in range(len(Color)):
        if Color[i] in lab:
            c=colors[i]
            
    m=''
    for i in range(len(Marker)):
        if Marker[i] in lab:
            m=markers[i]
            
    l='-'
    for i in range(len(Line)):
        if Line[i] in lab:
            l=lines[i]
    
    return c,m,l

def getkeys(allkeys,ands=None,ors=None):
    validkeys=[]
    if ands:
        for key in allkeys:
            ok=True
            for a in ands:
                if a not in key:
                    ok=False
            if ok:
                validkeys.append(key)
    if ors:
        for key in allkeys:
            for a in ands:
                if a in key:
                    validkeys.append(key)
    return validkeys


Base_Folder = R"C:\Users\Miguel\Work\BulgeTest\RESULTS"
plyfilenames=["20180511LP_2k.ply","disk2k.ply","20180511LP_5k.ply","disk5k.ply","20180511LP_10k.ply","disk10k.ply"]
savefile = pjoin(Base_Folder,"AllData.pkl")

casedefault = {"prestrain":False,
                # "load":2E-2,
                "c1MR":0.25, #mu = 0.5MPa
                "kbulk":100,
                "augLagrangian":True,
                "fibers":True,
                "Quadratic":True,
                "ThreeF":True,
                "fibertype":3,
                "dispersion":True,
                }



FibTypes = ["Uniform","Aligned","Average"]
MshTypes = ["2k","5k","10k"]
GeoTypes = ["disk","LP"]


RunModels = False
LoadModels = False


if LoadModels:

    Pressures = []
    Disps = []
    Labels = []

    for plyfilename in plyfilenames:
        SourcePLY = pjoin(Base_Folder,plyfilename)
        root,ext = os.path.splitext(plyfilename)

        if "disk" in plyfilename:
            casedefault = {**casedefault,"centerloc":(0.0,0.0,0.0)}
        else:
            casedefault = {**casedefault,"centerloc":(723,713,454)}


        if RunModels:
            Nodes,Elems = MAIO.PLYIO(SourcePLY).ExportMesh()
            FEBio = MAIO.FEBioIO()
            FEBio.ImportMesh(Nodes,Elems)

        Cases={
            "IE_Aligned":{**casedefault,"kip":0.0,"load":2E-2},
            "IE_Average":{**casedefault,"kip":0.184,"load":2E-2},
            "IE_Uniform":{**casedefault,"kip":0.5,"load":2E-2},
            "ME_Aligned":{**casedefault,"kip":0.0,"load":-2E-2,"shellBC":True},
            "ME_Average":{**casedefault,"kip":0.184,"load":-2E-2,"shellBC":True},
            "ME_Uniform":{**casedefault,"kip":0.5,"load":-2E-2,"shellBC":True},
            }

        destfolder = pjoin(Base_Folder,root)
        MATL.MakeNewDir(destfolder)


        for case in Cases:
            print(root + " " + case)
            casefolder = pjoin(destfolder,case)
            MATL.MakeNewDir(casefolder)
            febiofile = pjoin(casefolder,root+"_"+case+".feb")
            
            if RunModels:
                FEBio.SaveFile(febiofile,**Cases[case])
                print("    Running FEBio. Check logfile for details...")
                MATL.RunProgram('FEBio2 "'+febiofile+'"',f_print=False) 
                print("    FEBio Simulation finished.")


            logfile = os.path.splitext(febiofile)[0]+".log"
            with open(logfile, "r") as fp:
                lines=fp.readlines()

            Success = lines[-2].strip().split()[0]=="N"
            if Success:
                t,d = getdata(lines)
                t=np.insert(t,0,0.0)
                d=np.insert(d,0,0.0)

                t*=20
                if "IE" in case:
                    t*=-1

                Pressures.append(-t)
                Disps.append(-d)
                    
                Labels.append(root+" - "+case)

            else:
                print("SIMULATION DIDNT CONVERGE")


    AllData = {}
    for p,d,l in zip(Pressures,Disps,Labels):
        fiber = [fib for fib in FibTypes if fib in l][0]
        mesh = [msh for msh in MshTypes if msh in l][0]
        geo = [geo for geo in GeoTypes if geo in l][0]

        neg=False
        if p[-1]<0:
            neg=True
            p=np.fliplr([p])[0]
            d=np.fliplr([d])[0]

        datadict = {"fiber":fiber,"mesh":mesh,"geo":geo,"Press":p,"Disp":d}

        key = fiber + mesh + geo
        if key not in AllData.keys():
            AllData[key] = datadict
        else:
            if neg:
                tupd=(d[:-1],AllData[key]["Disp"])
                tupp=(p[:-1],AllData[key]["Press"])
            else:
                tupd=(AllData[key]["Disp"],d[1:])
                tupp=(AllData[key]["Press"],p[1:])

            AllData[key]["Disp"] =  np.hstack(tupd)
            AllData[key]["Press"] = np.hstack(tupp)
            AllData[key]["Compl"] = np.gradient(AllData[key]["Disp"],AllData[key]["Press"])
    save_obj(savefile,AllData)
else:
    AllData = load_obj(savefile)


def standardizePlot(ax,title,xlab,ylab,xlim=None,ylim=None,hline=True,legend=True,lblpressure=True,putimg=True,compliance=False):
    ax.axhline(linewidth=1,color='k',zorder=1)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)

    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    if legend: ax.legend(prop={'size': 8})

    if lblpressure:
        ax.text(.15, -.2, "(ME Pressure)", ha='center',transform = axs.transAxes)
        ax.text(.85, -.2, "(IE Pressure)", ha='center',transform = axs.transAxes)

    if putimg:
        if compliance:
            MEy=0.26
            IEy=0.26
        else:
            MEy=0.56
            IEy=0.44

        arr_img = plt.imread(pjoin(Base_Folder,"MembraneForPressureME.png"), format='png')
        imagebox = mplob.OffsetImage(arr_img, zoom=0.08)
        imagebox.image.axes = ax
        ab = mplob.AnnotationBbox(imagebox, (0.15,MEy), xycoords='axes fraction', pad=0.5, frameon=False)
        ax.add_artist(ab)

        arr_img = plt.imread(pjoin(Base_Folder,"MembraneForPressureIE.png"), format='png')
        imagebox = mplob.OffsetImage(arr_img, zoom=0.08)
        imagebox.image.axes = ax
        ab = mplob.AnnotationBbox(imagebox, (0.85,IEy), xycoords='axes fraction', pad=0.5, frameon=False)
        ax.add_artist(ab)



OUTF = pjoin(Base_Folder,"PlotResults")
MATL.MakeNewDir(OUTF)

plots="12345"


markersize=6
markevery=6


# Mesh Refinement
if "1" in plots:
    for geo in ["disk","LP"]:
        fig = plt.figure(figsize=MAFP.HalfColumn)
        axs  = fig.add_subplot(111)

        datakeys = getkeys(AllData.keys(),ands=["Uniform",geo])

        for key in datakeys:
            label= fixlabel(key,Mesh=True)
            c,m,l = getcml(key,Color=MshTypes,Marker=MshTypes)
            axs.plot(AllData[key]["Press"],AllData[key]["Disp"],label=label,color=c,marker=m,markevery=markevery,markersize=markersize,linestyle=l)

        geol="Disk" if geo=="disk" else "Membrane"

        standardizePlot(axs,"Effect of Mesh Refinement: "+geol+"\n with average fiber distribution",
                            "Pressure (kPa)",
                            "Displacement of control point ($\mu$m)",
                            xlim=[-20.0,20.0],ylim=[-250,250],
                            hline=True,legend=True)

        fig.savefig(pjoin(OUTF,"MeshRefinement_"+geol+".pdf"), bbox_inches='tight', dpi=600)
        # fig.show()
        # plt.waitforbuttonpress()
        plt.close(fig)


# Disk Vs Membrane (displacement)
if "2" in plots:
    fig = plt.figure(figsize=MAFP.HalfColumn)
    axs  = fig.add_subplot(111)

    datakeys = getkeys(AllData.keys(),ands=["Uniform","10k"])

    for key in datakeys:
        label= fixlabel(key,Geo=True)
        c,m,l = getcml(key,Color=GeoTypes,Marker=GeoTypes)
        axs.plot(AllData[key]["Press"],AllData[key]["Disp"],label=label,color=c,marker=m,markevery=markevery,markersize=markersize,linestyle=l)

    standardizePlot(axs,"Effect of geometry with\naverage fiber distribution",
                        "Pressure (kPa)",
                        "Displacement of control point ($\mu$m)",
                        xlim=[-20.0,20.0],ylim=[-250,250],
                        hline=True,legend=True)

    fig.savefig(pjoin(OUTF,"GeometryEffect_Displacement.pdf"), bbox_inches='tight', dpi=600)
    # fig.show()
    # plt.waitforbuttonpress()
    plt.close(fig)

# Disk Vs Membrane (compliance)
if "3" in plots:
    fig = plt.figure(figsize=MAFP.HalfColumn)
    axs  = fig.add_subplot(111)

    datakeys = getkeys(AllData.keys(),ands=["Uniform","10k"])

    for key in datakeys:
        label= fixlabel(key,Geo=True)
        c,m,l = getcml(key,Color=GeoTypes,Marker=GeoTypes)
        axs.plot(AllData[key]["Press"],AllData[key]["Compl"],label=label,color=c,marker=m,markevery=markevery,markersize=markersize,linestyle=l)

    standardizePlot(axs,"Effect of geometry with\naverage fiber distribution",
                        "Pressure (kPa)",
                        "Compliance of membrane at\ncontrol point ($\mu$m.kPa$^{-1}$)",
                        xlim=[-20.0,20.0],ylim=[0,250],
                        hline=False,legend=True,compliance=True)

    fig.savefig(pjoin(OUTF,"GeometryEffect_Compliance.pdf"), bbox_inches='tight', dpi=600)
    # fig.show()
    # plt.waitforbuttonpress()
    plt.close(fig)

# Fiber Alignment (Disp)
if "4" in plots:
    for geo in ["disk","LP"]:
        fig = plt.figure(figsize=MAFP.HalfColumn)
        axs  = fig.add_subplot(111)
        datakeys = getkeys(AllData.keys(),ands=["10k",geo])

        for key in datakeys:
            label= fixlabel(key,Fiber=True)
            c,m,l = getcml(key,Color=FibTypes,Marker=FibTypes)
            axs.plot(AllData[key]["Press"],AllData[key]["Disp"],label=label,color=c,marker=m,markevery=markevery,markersize=markersize,linestyle=l)

        
        geol="Disk" if geo=="disk" else "Membrane"

        standardizePlot(axs,"Effect of Fiber Alignment in\nthe Displacement for "+geol,
                            "Pressure (kPa)",
                            "Displacement of control point ($\mu$m)",
                            xlim=[-20.0,20.0],ylim=[-250,250],
                            hline=True,legend=True)

        fig.savefig(pjoin(OUTF,"FiberAlignment_"+geol+"_Disp.pdf"), bbox_inches='tight', dpi=600)
        # fig.show()
        # plt.waitforbuttonpress()
        plt.close(fig)


# Fiber Alignment (Compliance)
if "5" in plots:
    fig = plt.figure(figsize=MAFP.HalfColumn)
    axs  = fig.add_subplot(111)
    datakeys = getkeys(AllData.keys(),ands=["10k","LP"])

    for key in datakeys:
        label= fixlabel(key,Fiber=True)
        c,m,l = getcml(key,Color=FibTypes,Marker=FibTypes)
        axs.plot(AllData[key]["Press"],AllData[key]["Compl"],label=label,color=c,marker=m,markevery=markevery,markersize=markersize,linestyle=l)

    
    standardizePlot(axs,"Effect of Fiber Alignment in\nthe Complianceof the Membrane",
                        "Pressure (kPa)",
                        "Compliance of membrane at\ncontrol point ($\mu$m.kPa$^{-1}$)",
                        xlim=[-20.0,20.0],ylim=[0,60],
                        hline=True,legend=True,compliance=True)
    fig.savefig(pjoin(OUTF,"FiberAlignment_Compliance.pdf"), bbox_inches='tight', dpi=600)
    # fig.show()
    # plt.waitforbuttonpress()
    plt.close(fig)