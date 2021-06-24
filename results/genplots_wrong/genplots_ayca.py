import pandas as pd
import random 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns

data = pd.read_csv("metrics_ayca.csv")




bw_color_palette = sns.color_palette("blend:0,.8")
sns.set_palette(bw_color_palette)




# 3D plot of 3 variables

if True:

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    labels = np.sort(np.array(data["Method name"].unique()))
    zdata = data["Insertion/Deletion"]
    ax.set_zlabel('Insertion/Deletion')

    xdata = data["Smoothness"]
    ax.set_xlabel('Smoothness')

    ydata = data["Time(log)"]
    ax.set_ylabel('Time(log)')

    color = -1
    for label in labels:
        color = color + 1
        ax.scatter3D(xdata.loc[data["Method name"] == label], ydata.loc[data["Method name"] == label], zdata.loc[data["Method name"] == label], c=[color,color,color,color,color,color,color], cmap='Spectral', vmin=0, vmax=len(labels), label=str(label));

    ax.legend(title="Method", bbox_to_anchor=(0, 1), loc='upper right')

    fig.savefig("scatter3D.png", bbox_inches='tight', pad_inches=0.01)


# 3D plot of 3 variables, 1 color
if False:
    labels = np.sort(np.array(data["Method name"].unique()))
    for label in labels:
        data_filtered = data.loc[data["Method name"] == label]
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlabel('Insertion/Deletion')
        ax.set_ylabel('Smoothness')
        ax.set_zlabel('Time(log)')

        ax.scatter3D(data_filtered["Insertion/Deletion"], data_filtered["Smoothness"], data_filtered["Time(log)"], color=(.5, .5, .5),alpha=.5);

        fig.savefig("scatter3D_ayca.png")



if False:
    # 2D plot of 2 variables

    fig = plt.figure()
    ax = plt.axes()

    labels = np.sort(np.array(data["Class label"].unique()))
    xdata = data["Compactness"]
    ax.set_xlabel('Compactness')

    ydata = data["Correctness"]
    ax.set_ylabel('Correctness')

    for label in labels:
        ax.scatter(xdata.loc[data["Class label"] == label], ydata.loc[data["Class label"] == label], c=data.loc[data["Class label"] == label]["Class label"], cmap='Spectral', vmin=labels[0], vmax=labels[-1], label="Class "+str(label));
        
    ax.legend(title="Class")

    fig.savefig("scatter2D.png")







if False:
    # 2D plot of 2 variables 1 color

    fig = plt.figure()
    ax = plt.axes()

    xdata = data["Compactness"]
    ax.set_xlabel('Compactness')

    ydata = data["Correctness"]
    ax.set_ylabel('Correctness')

    ax.scatter(xdata, ydata, color=(.5, .5, .5),alpha=.5);

    fig.savefig("scatter2D2.png")




if False:

    # 2D plot of 2 variables with histograms https://seaborn.pydata.org/examples/regression_marginals.html

    fig = plt.figure()
    ax = plt.axes()

    g = sns.jointplot(data=data.loc[data["Method name"] == "Vanilla Saliency"], x="Compactness", y="Correctness", color=".5", ) # , hue="Method name"
    g.plot_joint(sns.kdeplot, color=".8")
    g.plot_marginals(sns.rugplot, color=".5", height=-.15, clip_on=False)
                      
    plt.savefig("jointplot.png")








if False:

    # Violinplots of methods

    metricname = "Correctness"
    methodnames = data["Method name"].unique()

    print(methodnames)

    data_to_plot = [data.loc[data['Method name'] == methodnames[0]][metricname], data.loc[data['Method name'] == methodnames[1]][metricname], data.loc[data['Method name'] == methodnames[2]][metricname], data.loc[data['Method name'] == methodnames[3]][metricname], data.loc[data['Method name'] == methodnames[4]][metricname], data.loc[data['Method name'] == methodnames[5]][metricname]]
    fig = plt.figure()
    ax = plt.axes()

    ax.violinplot(data_to_plot, showmeans=True, showmedians=False,showextrema=True );

    ax.yaxis.grid(True)
    ax.set_xticks([y+1 for y in range(len(data_to_plot))])
    ax.set_xticklabels(methodnames)
    ax.set_ylabel(metricname)
      
    fig.savefig("violinplot.png")



# Violinplots of seaborn https://seaborn.pydata.org/examples/grouped_violinplots.html

labels = ["Insertion/Deletion","Smoothness","Time(log)"]

for metricname in labels:

    fig = plt.figure(figsize=(12,3))
    ax = plt.axes()

    methodnames = data["Method name"].unique()

    data_to_plot = [data.loc[data['Method name'] == methodnames[0]][metricname], data.loc[data['Method name'] == methodnames[1]][metricname], data.loc[data['Method name'] == methodnames[2]][metricname], data.loc[data['Method name'] == methodnames[3]][metricname], data.loc[data['Method name'] == methodnames[4]][metricname], data.loc[data['Method name'] == methodnames[5]][metricname], data.loc[data['Method name'] == methodnames[6]][metricname]]

    sns.violinplot( data=data_to_plot, orient="v", inner="quartile", color=".8")
    sns.stripplot( data=data_to_plot, jitter=.05,  color="0" ,size=2, alpha=0.2, orient="v")

    ax.set_xticks([y for y in range(len(data_to_plot))])
    ax.set_xticklabels(methodnames)
    ax.set_ylabel(metricname)
    ax.yaxis.grid(True)

    plt.savefig("violinplot_"+metricname.replace("/","")+".png")



exit()


# 2D plot of all variables with histograms https://seaborn.pydata.org/examples/regression_marginals.html
labels = np.sort(np.array(data["Method name"].unique()))

for label in labels:
    data_filtered = data.loc[data["Method name"] == label]

    fig = plt.figure()
    ax = plt.axes()
    sns.heatmap(data_filtered[["Completeness","Compactness","Correctness","Symbolism", "f1","Cropping","MoRF","Ssim","Smoothness"]].corr().round(2), square=True, cmap='RdYlGn', annot=True)
    plt.savefig("heatmap_"+label+".png", bbox_inches='tight', pad_inches=0.01)

fig = plt.figure()
ax = plt.axes()
sns.heatmap(data[["Completeness","Compactness","Correctness","Symbolism", "f1","Cropping","MoRF","Ssim","Smoothness"]].corr().round(2), square=True, cmap='RdYlGn', annot=True)
plt.savefig("heatmap_general.png", bbox_inches='tight', pad_inches=0.01)




# 2D plot of all variables with histograms https://seaborn.pydata.org/examples/regression_marginals.html


data_filtered = data

fig = plt.figure()
ax = plt.axes()
sns.pairplot(data=data_filtered[["Completeness","Compactness","Correctness","Symbolism", "f1","Cropping","MoRF","Ssim","Smoothness"]], height=1.4, kind="kde") #, ,"f1"

plt.savefig("pairplot_general.png")



labels = np.sort(np.array(data["Method name"].unique()))

for label in labels:
    
    data_filtered = data.loc[data["Method name"] == label]

    fig = plt.figure()
    ax = plt.axes()
    sns.pairplot(data=data_filtered[["Completeness","Compactness","Correctness","Symbolism", "f1","Cropping","MoRF","Ssim","Smoothness","Method name"]], height=1.4, kind="kde") #, ,"f1"

    plt.savefig("pairplot_"+label+".png")




