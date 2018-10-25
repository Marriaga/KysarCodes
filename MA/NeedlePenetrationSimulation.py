import numpy as np
import scipy.signal as spsig
import scipy.optimize as spopt
import matplotlib.pyplot as plt
import os
import MA.Tools as MATL
import MA.FigureProperties as MAFP

# FN = FH*c + FV*s
# FT* = FV*c - FH*s
# FH = Kh*dh
# FV = Kv*(dv+dp)

class ComputePenetration(object):
    def __init__(self,Horizontal_Stiffness,Vertical_Stiffness,Friction_Coef_Static,Friction_Coef_Dynamic,Depth=100,Angle=30,dh0=0.0):

        #TODO: FIX INITIALIZATION OF PROBLEM TO TAKE DH0

        self.kh = Horizontal_Stiffness
        self.kv = Vertical_Stiffness
        self.us = Friction_Coef_Static
        self.ud = Friction_Coef_Dynamic
        self.Depth = Depth
        self.dh0 = dh0
        self.theta = np.cos(np.radians(Angle)),np.sin(np.radians(Angle))
        self.thick = Depth*self.theta[1]/self.theta[0]
        
        self.slipping = False
        self.ctime = 0
        self.Dt = 0.001
        self.numericalviscosity = 0.1
        self.cdx = np.array([0.0,0.0])

    def get_FT(self,FT_pred,FN):
        FTS = self.us*FN
        FTD = self.ud*FN

        if not self.slipping and abs(FT_pred)>FTS:
            #Force greater than static slipping
            self.slipping = True
        elif self.slipping and abs(FT_pred)<=FTD:
            #Force smaller than dynamic slipping
            self.slipping = False
        
        if self.slipping:
            return np.sign(FT_pred)*FTD
        else:
            return FT_pred

    def get_theta(self,dv,init=False):
        if not init:
            cc,cs = self.get_theta(self.cdx[1],init=True)

        if -dv >= self.Depth:
            c,s = 1.0 , 0.0
        else:
            c,s = self.theta

        if init:
            return c,s
        else:
            fff=0.0
            return cc*(1-fff)+c*fff,cs*(1-fff)+s*fff

    def dhfdv(self,dv):
        if -dv >= self.Depth:
            return self.thick
        elif -dv <= 0:
            return 0.0
        else:
            return -dv*self.theta[1]/self.theta[0]

    @staticmethod
    def Force(k,d,m=0.02,stype="linear"):
        if stype=="linear":
            return k*d
        elif stype == "exponential":
            return d*k*m*np.exp(m*abs(d))
        elif stype == "exp2":
            return d*0.003*0.05*np.exp(0.05*abs(d))
        elif stype == "bilin":
            print(d)
            if abs(d)<200:
                return k*d
            else:
                return k*20+k/2*np.sign(d)*(abs(d)-20)

    def residual(self,dx):
        dh,dv = dx
        #dh=self.dhfdv(dv)-self.dh0
        cdh,cdv = self.cdx
        #cdh = self.dhfdv(cdv)-dh0
        FH = max(self.Force(self.kh,dh,stype="linear"),0.0)
        FV = self.Force(self.kv,dv+self.cdp,stype="linear")
        c,s = self.get_theta(dv)

        FN = FH*c + FV*s
        FT_pred =  FV*c - FH*s

        FT = self.get_FT(FT_pred,FN)

        #rh = -self.numericalviscosity*(dh-cdh)/self.Dt - FH + FN*c - FT*s
        #rh = -self.numericalviscosity*(dh-cdh)/self.Dt + dh - self.dhfdv(dv)
        #rh = 0.0
        rh = -self.numericalviscosity*(dh-cdh)/self.Dt + (dh-self.dhfdv(dv)+self.dh0)
        rv = -self.numericalviscosity*(dv-cdv)/self.Dt - FV + FN*s + FT*c

        return np.array([rh,rv])
    

def loadfunction(i,N,transition_fraction,loadtype="roundtrip"):
    current_fraction = i/N

    if loadtype=="roundtrip":
        current_fraction*=2
        if current_fraction>1.0: current_fraction=2-current_fraction
    
    return min(current_fraction/transition_fraction,1.0)

def RunSimulaion(plot=False,opath=None):
    if plot==True and opath is None:
        raise ValueError("PLEASE SET opath TO STORE FIGURES")
        
    scaling = 180
    Horizontal_Stiffness = 8/scaling
    Vertical_Stiffness = 2.2/scaling
    angvisc=49
    Friction_Coef_Static = np.tan(np.radians(angvisc))
    Friction_Coef_Dynamic = np.tan(np.radians(angvisc))
    Depth=100
    Angle=20
    dh0 = 25

    MP = ComputePenetration(Horizontal_Stiffness,Vertical_Stiffness,Friction_Coef_Static,Friction_Coef_Dynamic,Depth=Depth,Angle=Angle,dh0=dh0)
    MP.Dt = 1.2
    MP.numericalviscosity = 0.000001

    N_steps=1000
    Np = N_steps+1
    fraction = 0.9

    max_penetration = 270

    MembranePosition = np.empty(Np)
    MembranePositionH = np.empty(Np)
    PenetratorPosition = np.empty(Np)
    VerticalForce = np.empty(Np)
    Time = np.empty(Np)

    for ti in range(Np):
        MP.cdp = max_penetration*max(0.0,loadfunction(ti,N_steps,fraction,loadtype="roundtrip"))
        method = ["hybr","lm","broyden1","krylov"][0]
        # result = spopt.fsolve(MP.residual, MP.cdx)
        result = spopt.root(MP.residual, MP.cdx, method=method)
        # result = spopt.root(MP.residual, MP.cdx)

        dx=result.x
        MP.cdx = dx
        MP.ctime += MP.Dt

        MembranePosition[ti] = dx[1]
        MembranePositionH[ti] = MP.dhfdv(dx[1])
        PenetratorPosition[ti] = MP.cdp
        VerticalForce[ti] = MP.kv*(MP.cdp+dx[1])
        Time[ti]=MP.ctime-MP.Dt

    if plot:
        figs, axs = plt.subplots(2)
        ax0=axs[0]
        ax0.plot(Time, PenetratorPosition, '-', label='Needle Position')
        ax0.plot(Time, -MembranePosition, ':', label='Membrane Position')
        ax0.set_xlabel('"Time"')
        ax0.set_ylabel('Displacement (µm)')
        ax0.legend()       
        ax0.grid()     

        ax1=axs[1]
        ax1.plot(Time, VerticalForce, '-k', label='Vertical Force (mN)')
        ax1.set_xlabel('"Time"')
        ax1.set_ylabel('Vertical Force (mN)')
        ax1.grid()

        figs.subplots_adjust(hspace=.5)
        figs.savefig(opath + "_Position.png",dpi = 600)
        plt.close(figs)


        figs, axs = plt.subplots(figsize=MAFP.Half_p)
        ax2=axs
        ax2.plot(PenetratorPosition, VerticalForce, '-', label='Force vs Displacement')
        ax2.set_xlabel('Needle Displacement (µm)')
        ax2.set_ylabel('Vertical Force (mN)')
        ax2.grid()     

        figs.savefig(opath + "_FD.png",dpi = 600)
        plt.close(figs)

    return PenetratorPosition, VerticalForce

def readfile(filepath):
    X=[]
    Y=[]
    with open(filepath,"r") as fp:
        fp.readline()
        for line in fp:
            Dat=line.strip().split()
            X.append(float(Dat[0]))
            Y.append(float(Dat[1]))
    return np.array(X),np.array(Y)

def ImportDataCycles(filepath,Xcut=60,Ycut=0.1,plot=False,opath=None):
    if opath is None:
        opath, _ = os.path.splitext(filepath)
    X,Y = readfile(filepath)

    # Normalize Position and Force
    L=0
    while X[L]==X[0]: L+=1
    zeroX = np.average(X[:L])
    zeroY = np.average(Y[:L])
    X-=zeroX
    Y-=zeroY

    # Cleanup - Remove Values<0 and add start and end at 0
    II=X>0
    II[0]=II[-1]=True
    X=X[II]
    Y=Y[II]
    X[0]=X[-1]=Y[0]=Y[-1]=0.0


    #MATL.QPlot(X,Y)

    #X,Y = reduce_points(X,Y)

    # Get Points with low X and low Y
    Ilist=np.arange(len(X))
    Ilist=Ilist[X<Xcut] #Low X
    Ilist=Ilist[np.absolute(Y[Ilist])<Ycut] # Low Y

    # Get Cycle Start/End
    SElist=[Ilist[0]]
    for i in range(len(Ilist)-1):
        if Ilist[i+1]- Ilist[i]>20:
            SElist.append(Ilist[i+1])
        elif X[Ilist[i+1]]- X[Ilist[i]]<=0:
            SElist[-1]=Ilist[i+1]

    Xlist=[]
    Ylist=[]
    for i in range(len(SElist)-1):
        Xlist.append(np.array(X[SElist[i]:SElist[i+1]]))
        Ylist.append(np.array(Y[SElist[i]:SElist[i+1]]))

    if plot:
        figs, axs = plt.subplots(1)
        ax0=axs
        for i in range(len(Xlist)):
            ax0.plot(Xlist[i], Ylist[i], '-')
        ax0.set_xlabel('Needle Displacement (µm)')
        ax0.set_ylabel('Vertical Force (mN)')     
        ax0.grid()     
        figs.savefig(opath + "_RealData.png")
        plt.close(figs)

    return Xlist,Ylist


def CompareDataWithSim(ZReal,FReal,ZSimu,FSimu,opath):
    figs, axs = plt.subplots(1)
    ax0=axs
    ax0.plot(ZSimu, FSimu, '-', label='Simulation')
    ax0.plot(ZReal, FReal, ':', label='Real Data')
    ax0.set_xlabel('Needle Displacement (µm)')
    ax0.set_ylabel('Vertical Force (mN)')
    ax0.legend()       
    ax0.grid()     
    figs.savefig(opath + "_ComparisonSimReal.png")
    plt.close(figs)


def reduce_points(x,y,x_increment="auto"):
    Range_X = np.amax(x)-np.amin(x)
    Range_Y = np.amax(y)-np.amin(y)

    if type(x_increment)==type("") and x_increment=="auto":
        x_increment = Range_X/200
    
    y_increment = Range_Y*x_increment/Range_X

    x_new = [x[0]]
    y_new = [y[0]]

    for i in range(len(x)):
        if abs(x[i]-x_new[-1])>= x_increment or abs(y[i]-y_new[-1])>= y_increment:
            x_new.append(x[i])
            y_new.append(y[i])

    return np.array(x_new),np.array(y_new)

def ComputeFractureEnergy(Zlist, Flist,SurfaceArea,SIScaling=1,PostPerfCyclesIndex=None):
    Energy=[]
    for i in range(len(Zlist)):
        Energy.append(np.trapz(Flist[i],Zlist[i]))

    Energy = np.array(Energy)

    if PostPerfCyclesIndex is None or PostPerfCyclesIndex == 0:
        EnergyDiff = Energy[0]-np.average(Energy[1:])
    else:
        EnergyDiff = Energy[0]-np.average(Energy[PostPerfCyclesIndex])

    print("DifRatio",np.average(Energy[1:])/Energy[0])

    FractureEnergy=EnergyDiff/SurfaceArea*SIScaling
    return FractureEnergy

def MAINANALYSIS(filepath):
    opath, _ = os.path.splitext(filepath)
    Zlist, Flist = ImportDataCycles(filepath,plot=True,opath=opath)
    #ZSimu, FSimu = RunSimulaion(plot=True,opath=opath)
    #CompareDataWithSim(Zlist[1], Flist[1], ZSimu, FSimu,opath=opath)
    FractureEnergy = ComputeFractureEnergy(Zlist, Flist,100*30*2,SIScaling=1000,PostPerfCyclesIndex=1)
    print(FractureEnergy)

### MAIN ###

DataFolder="C:\\Users\\Miguel\\Work\\RWM-Energy Release Rate\\20180907\\data\\M2"
Files = ["FD1.txt","FD2.txt","FD3.txt"]
# DataFolder="C:\\Users\\Miguel\\Work\\RWM-Energy Release Rate\\20180907\\data\\M1"
# Files = ["Fd.txt"]

for thefile in Files:
    filepath = os.path.join(DataFolder,thefile)
    MAINANALYSIS(filepath)
