
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import statsmodels.formula.api as smf
import itertools
import numpy as np
import copy

import MA.to_precision as tp  # From https://bitbucket.org/william_rusnack/to-precision


############## Begin hack for markers in swarm ##############
from matplotlib.axes._axes import Axes
import matplotlib.markers as mmarkers
from seaborn import color_palette

def GetColor2Marker(markers):
    palette = color_palette()
    mkcolors = [(palette[i]) for i in range(len(markers))]
    return dict(zip(mkcolors,markers))

def fixlegend(ax,markers,markersize=8,**kwargs):
    _,l = ax.get_legend_handles_labels()
    palette = color_palette()
    mkcolors = [(palette[i]) for i in range(len(markers))]
    newHandles = [plt.Line2D([0],[0], ls="none", marker=m, color=c, mec="none", markersize=markersize,**kwargs) \
                for m,c in zip(markers, mkcolors)]
    ax.legend(newHandles,l)

old_scatter = Axes.scatter
def new_scatter(self, *args, **kwargs):
    colors = kwargs.get("c", None)
    co2mk = kwargs.pop("co2mk",None)
    FinalCollection = old_scatter(self, *args, **kwargs)
    if co2mk is not None and isinstance(colors, np.ndarray):
        Color2Marker = GetColor2Marker(co2mk)
        paths=[]
        for col in colors:
            mk=Color2Marker[tuple(col)]
            marker_obj = mmarkers.MarkerStyle(mk)
            paths.append(marker_obj.get_path().transformed(
                        marker_obj.get_transform()))
        FinalCollection.set_paths(paths)
    return FinalCollection
Axes.scatter = new_scatter

############## End hack. ##############



def setupPaperStyle():
    sns.set_palette("deep")
    #plt.rcParams['text.usetex']=True
    #plt.rcParams['text.latex.preamble']=r'\usepackage{amsmath} \boldmath'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['lines.markersize'] = 9
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['font.size'] = 10

    # {'font.size': 10.0, 'axes.labelsize': 12.0, 'axes.titlesize': 12.0, 'xtick.labelsize': 11.0, 'ytick.labelsize': 11.0,
    # 'legend.fontsize': 11.0, 'axes.linewidth': 1.25, 'grid.linewidth': 1.0, 'lines.linewidth': 1.5, 'lines.markersize': 5.0,
    # 'patch.linewidth': 1.0, 'xtick.major.width': 1.25, 'ytick.major.width': 1.25, 'xtick.minor.width': 1.0, 'ytick.minor.width': 1.0,
    # 'xtick.major.size': 6.0, 'ytick.major.size': 6.0, 'xtick.minor.size': 4.0, 'ytick.minor.size': 4.0}

    # print(sns.plotting_context())

MKS = ['o', '^', 's','D','v',"d", '*', 'P','H','X']

### FIGURE SIZES ###
# In Matplotlib, figure sizes are tuples with dimensions in inches (x,y)
class PaperType(object):
    def __init__(self,Name,LongEdge,ShortEdge,units):
        self.Name = Name
        self.LongEdge = LongEdge
        self.ShortEdge = ShortEdge
        self.units = units
    def getHV(self,Ori,Margins):
        if Ori == "P":
            V=self.ShortEdge
            H=self.LongEdge
        elif Ori == "L":
            V=self.LongEdge
            H=self.ShortEdge
        else:
            raise ValueError("Wrong type of orientation. Ori = 'P' or 'L'")
        return V-2*Margins[1],H-2*Margins[0]

#Types of PaperSizes
PaperObjects=[]
PaperObjects.append(PaperType("Letter",11.0,8.5,"in"))

PaperObjects={obj.Name:obj for obj in PaperObjects}

def get_size(Nrow,Ncol,Ori="P",Paper="Letter",Margins=(1.0,1.0)):
    PaperObj = PaperObjects[Paper]
    V,H = PaperObj.getHV(Ori,Margins)
    return V/Ncol,H/Nrow


HalfPage=get_size(2,1,Ori="P",Paper="Letter",Margins=(1.0,1.0))



### PLOTTING FUNCTIONS ###

def setxaxis(ax,label=None,limits=None,step=None):
    if label is not None: ax.set_xlabel(label)
    if limits is not None: ax.set_xlim(limits)
    if step is not None: ax.xaxis.set_major_locator(ticker.MultipleLocator(step))

def setyaxis(ax,label=None,limits=None,step=None):
    if label is not None: ax.set_ylabel(label)
    if limits is not None: ax.set_ylim(limits)
    if step is not None: ax.yaxis.set_major_locator(ticker.MultipleLocator(step))



# def new_box_plot_columns(df,categories_column,list_of_columns,legend_title,linewidth=0.7,order=None,**boxplotkwargs):
#     columns = [categories_column] + list_of_columns
#     newdf = df[columns].copy()
#     data = newdf.melt(id_vars=[categories_column], var_name=legend_title, value_name="Value")

#     g = sns.FacetGrid(data, col=categories_column,col_order=order,sharex=False)
#     g.map(sns.boxplot,categories_column,"Value",legend_title,linewidth=linewidth)

#     return plt.gcf(), np.array(g.axes.flat)

def box_plot_columns(df,categories_column,list_of_columns,legend_title,y_axis_title,linewidth=0.7,**boxplotkwargs):
    columns = [categories_column] + list_of_columns
    newdf = df[columns].copy()
    data = newdf.melt(id_vars=[categories_column], var_name=legend_title, value_name=y_axis_title)
    return sns.boxplot(data=data, x=categories_column, y=y_axis_title, hue=legend_title, linewidth=linewidth, **boxplotkwargs)

def swarm_plot_columns(df,categories_column,list_of_columns,legend_title,y_axis_title,**swarmplotkwargs):
    columns = [categories_column] + list_of_columns
    newdf = df[columns].copy()
    data = newdf.melt(id_vars=[categories_column], var_name=legend_title, value_name=y_axis_title)
    return sns.swarmplot(data=data, x=categories_column, y=y_axis_title, hue=legend_title, **swarmplotkwargs)

## Full figures

def box_plot_figure(Data,categories_column,columns,column_labels=None,
    legend_title=" ",xlabel=None,ylabel=None,title=None,addHyperCyl=False,
    savepath=None,zeroline=False,useSwarm=False,co2mk=None):
    
    showlegend=True
    if legend_title is None:
        legend_title=" "
        showlegend=False

    newdf = Data[[categories_column] + columns].copy()
    if column_labels is None:
        column_labels = columns
    else:
        newdf = newdf.rename(columns=dict(zip(columns, column_labels)))

    fig, ax = plt.subplots(1,1,figsize=HalfPage)
    if useSwarm:
        if len(co2mk) != len(columns):
            raise ValueError("co2mk must have the same length as the number of columns in the hue.")
        ax = swarm_plot_columns(newdf,categories_column,column_labels,legend_title,"Values",ax=ax,order=["Bottom","Left", "Center", "Right", "Top"],co2mk=co2mk)
    else:    
        ax = box_plot_columns(newdf,categories_column,column_labels,legend_title,"Values",ax=ax,order=["Bottom","Left", "Center", "Right", "Top"])

    # Fix Legend
    legtitle =  ax.get_legend().get_title().get_text()
    if useSwarm and co2mk is not None:
        fixlegend(ax,co2mk)
    
    leg = ax.get_legend()
    leg.set_title(legtitle)
    leg._legend_box.align = "left"
    title_inst = leg.get_title()
    title_inst.set_fontweight('bold')
    if not showlegend:
        leg.remove()

    if zeroline: ax.axhline(color="k",zorder=0)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title)

    if addHyperCyl:
        arrowprops = dict(arrowstyle='<->',linewidth=4,mutation_scale=25)
        
        ax.annotate("", (0.05,0.5), xytext=(0.55,0.5),  xycoords =ax.transAxes,arrowprops=arrowprops)
        ax.text(0.3, 0.6, "Hyperbolic Type", transform=ax.transAxes, fontsize=14, va="top", ha="center")

        ax.annotate("", (0.65,0.8), xytext=(0.95,0.8),  xycoords =ax.transAxes,arrowprops=arrowprops)
        ax.text(0.8, 0.9, "Cylindrical Type", transform=ax.transAxes, fontsize=14, va="top", ha="center")


    if savepath is None:
        plt.show()
    else:
        fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)



def regression_figure(df,x,y,line45=True,xlabel=None,ylabel=None,title=None,savepath=None,xlims=[0,45],ylims=[0,60],xstep=15,ystep=15):
    # Open Figure
    fig, ax = plt.subplots(1,1,figsize=HalfPage)
    ax = sns.scatterplot(x=x, y=y, data=df,hue="Position",style="Position",ax=ax,markers=MKS)
    setxaxis(ax,x,xlims,xstep)
    setyaxis(ax,y,ylims,ystep)
    if line45: ax.plot([0,45],[0,45],color="k")
    ax = sns.regplot(x=x, y=y, data=df, truncate=False, scatter=False,ax=ax,line_kws={"linestyle":"--"})

    # Improve Labels
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title)

    # Compute Linear Regression
    LinRegRes = smf.ols(y+" ~ "+x,df).fit()
    pvalue = tp.sci_notation(LinRegRes.pvalues[x],3)
    rsquared = tp.std_notation(LinRegRes.rsquared,3) # pylint: disable=E1101

    # Place box with regression properties
    textstr = "p-value = " + pvalue + "\nR$^2$ = " + rsquared
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.5, 0.95, textstr, transform=ax.transAxes, fontsize=14, va="top", ha="center",
    verticalalignment='top',bbox=props)
    
    # Fix Legend Title
    leg = ax.get_legend()
    leg._legend_box.align = "left"
    title_inst = leg.get_title()
    title_inst.set_fontweight('bold')

    if savepath is None:
        plt.show()
    else:
        fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)



def polar_figure(Angles,Values,ScatterPoints=None,xlabel=None,ylabel=None,title=None,savepath=None,xlims=[0,np.pi],ylims=None,xstep=np.pi/8,ystep=None):

    # Figure with angle of Max RSquared
    fig, ax = plt.subplots(1,1,subplot_kw={"projection":"polar"})
    ax.plot(np.radians(Angles),Values,lw=2.0)
    ax.scatter(np.radians([ScatterPoints[0]]),[ScatterPoints[1]],color='r',zorder=3)
    setxaxis(ax,ylabel,xlims,xstep)
    setyaxis(ax,xlabel,ylims,ystep)
    if title is not None: ax.set_title(title)
    #if xlabel is not None: ax.set_ylabel(xlabel, labelpad=30)
    ax.xaxis.set_label_coords(0.75, 0.18)

    #if ylabel is not None: ax.set_xlabel(ylabel, labelpad=-50)
    ax.yaxis.set_label_coords(-0.1, 0.5)
    
    if savepath is None:
        plt.show()
    else:
        fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)
