import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns



names=['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
       'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight',
       'engine-type', 'num-of-cylinders', 'engine-style', 'fuel-system', 'bore', 'stroke',
       'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price'] #naming dataframe columns

data = pd.read_csv('imports-85.data', names=names).drop(columns=['normalized-losses', 'fuel-type',
                  'aspiration', 'num-of-doors', 'body-style', 'engine-location',
                  'wheel-base', 'length', 'width', 'height', 'curb-weight',
       'engine-type', 'num-of-cylinders', 'engine-style', 'fuel-system', 'bore',
       'stroke', 'compression-ratio', 'peak-rpm', 'city-mpg', 'highway-mpg',]) #stripping unneeded dataframe columns

tlist = []
for i in range(data.shape[0]): #getting rid of data that can't be used,
    for j in data.columns: #sadly this removes our only renault sample(s)
        if data[j][i] == '?':
            tlist.append(i)
            
tlist.sort()
k = 0
for i in tlist:
    data = data.drop(data.index[i-k])
    k+=1

data = data.reset_index()
        
data['symboling'] = pd.to_numeric(data['symboling'])
data['horsepower'] = pd.to_numeric(data['horsepower'])
data['price'] = pd.to_numeric(data['price'])

makes = ['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda',
         'isuzu', 'jaguar', 'mazda', 'mercedes-benz', 'mercury',
         'mitsubishi', 'nissan',' peugot', 'plymouth', 'porsche',
         'saab', 'subaru', 'toyota', 'volkswagen', 'volvo']

colours=['black', 'darkgray', 'lightgray', 'indianred', 'darkred', 'saddlebrown',
         'darkorange', 'gold', 'olivedrab', 'chartreuse', 'lightgreen',
         'darkgreen', 'springgreen', 'lightseagreen', 'cadetblue', 'dodgerblue',
         'royalblue', 'darkblue', 'blueviolet', 'mediumorchid', 'magenta']

drive = ['4wd', 'fwd', 'rwd']

markers = ['o', '>', '<']

plt.figure(figsize=(11,9))

for i in range(data.shape[0]):
    temp = data['make'][i]
    k = 0
    for j in makes:
        if temp == j:
            c = colours[k]
            break
        k+=1
        
    temp = data['drive-wheels'][i]    
    k = 0
    for j in drive:
        if temp == j:
            m = markers[k]
            break
        k+=1
    
    
    
    plt.scatter(data['price'][i], data['horsepower'][i], s=(1.5*(data['symboling'][i]+4))**2, c=c, marker=m, label=data['make'][i])

plt.legend()
hand, labl = plt.gca().get_legend_handles_labels()
plt.legend(np.unique(labl))
leg = plt.gca().get_legend()
tlist = []
for i in range(len(np.unique(labl))):
    tlist.append(mlines.Line2D([],[], color=colours[i], marker='o', linestyle='None', markersize=5, label=makes[i]))

plt.legend(handles=tlist, loc=2)
plt.xlabel('Price of car [$]', size='x-large')
plt.ylabel('Horsepower [hp]', size='x-large')
plt.title("""Correlation between manufacturer, cost, horsepower and
the drive vs the "riskyness" of a vehicle""", size='xx-large')
plt.annotate("""$\\blacktriangleleft=RWD$
$\u25cf=4WD$
$\\blacktriangleright=FWD$""", xy=(-12, -12), xycoords='axes points',
            size=14, ha='right', va='top',
            bbox=dict(boxstyle='round', fc='w'))
plt.annotate("Marker size = Riskyness", xy=(175, -25), xycoords='axes points',
            size=14, ha='right', va='top',
            bbox=dict(boxstyle='round', fc='w'))
plt.show()

data['drive-wheels'] = data['drive-wheels'].replace(to_replace='rwd', value=1)
data['drive-wheels'] = data['drive-wheels'].replace(to_replace='4wd', value=0)
data['drive-wheels'] = data['drive-wheels'].replace(to_replace='fwd', value=2)



fig, axs = plt.subplots(1, 3, figsize=(15,5), sharey=True)
sns.kdeplot(data, ax=axs[0], x='price', y='symboling', legend=False, fill=True, alpha=0.5, warn_singular=False)
sns.scatterplot(data, ax=axs[0], x='price', y='symboling', hue='make', palette=colours, legend=False)
sns.rugplot(data, ax=axs[0], x='price', hue='make', palette=colours, legend=False)
sns.kdeplot(data, ax=axs[1], x='horsepower', y='symboling', legend=False, fill=True, alpha=0.5, warn_singular=False)
sns.scatterplot(data, ax=axs[1], x='horsepower', y='symboling', hue='make', palette=colours, legend=False)
sns.rugplot(data, ax=axs[1], x='horsepower', hue='make', palette=colours, legend=False)
meanData = data.groupby(data['make']).mean()
sns.kdeplot(data, ax=axs[2], x='drive-wheels', y='symboling', legend=False, fill=True, alpha=0.5, warn_singular=False)
sns.scatterplot(meanData, ax=axs[2], x='drive-wheels', y='symboling', hue='make', palette=colours, legend=False)
fig.legend(handles=tlist, loc='center right', bbox_to_anchor=(1,0.5), ncol=1, fancybox=True, shadow=True)
fig.suptitle("""Correlation between manufacturer, cost, horsepower and
the drive vs the "riskyness" of a vehicle""", size='xx-large')
plt.xticks([0,1,2], ['4WD', 'RWD', 'FWD'])
axs[0].set(xlabel='Price of car [$]')
axs[0].set(ylabel='Symboling')
axs[1].set(xlabel='Horsepower [hp]')
axs[2].set(xlabel='Drive')
plt.sca(axs[0])
plt.yticks([-2,-1,0,1,2,3], [-2,-1,0,1,2,3])
plt.show()


#plt.figure(figsize=(11,9))
#meanData = data.groupby(data['make']).mean()
#meanData['symboling'] += 2 #sizes can't be negative
#meanData['drive-wheels'] = (meanData['drive-wheels']/1).round().astype(int)*1 #rounding to nearest int
#sns.scatterplot(meanData, x='price', y='horsepower', hue='make', palette=colours, s=100*meanData['symboling'],
#                style='drive-wheels', markers=['<', 'o', '>'], legend=False)
#plt.legend(handles=tlist,loc='upper left')
#plt.xlabel('Price of car [$]', size='x-large')
#plt.ylabel('Horsepower [hp]', size='x-large')
#plt.title("""Correlation between manufacturer and the mean cost, horsepower and
#the drive vs the "riskyness" of a vehicle""", size='xx-large')
#plt.annotate("""$\\blacktriangleleft=RWD$
#$\u25cf=4WD$
#$\\blacktriangleright=FWD$""", xy=(-12, -12), xycoords='axes points',
#            size=14, ha='right', va='top',
#            bbox=dict(boxstyle='round', fc='w'))
#plt.annotate("Marker size = Riskyness", xy=(175, -25), xycoords='axes points',
#            size=14, ha='right', va='top',
#            bbox=dict(boxstyle='round', fc='w'))
#plt.show()
