import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier
import matplotlib.pyplot as plt
from pygam import LogisticGAM



#---TASK 1---
df = pd.read_csv('PimaIndiansDiabetes.csv', sep=';', index_col=False)
tlist = []
for i in df['mass']:
    i = float(i.replace(',', '.'))
    tlist.append(i)    
df['mass'] = tlist

tlist = []
for i in df['diabetes']:
    if i == 'neg':
        i = 0
    else:
        i = 1
    tlist.append(i)

df['diabetes'] = tlist

tlist = []
for i in df['pedigree']:
    i = float(i.replace(',', '.'))
    tlist.append(i)
df['pedigree'] = tlist

df = df.iloc[: , 1:] #remove extra index

X = df[['pregnant', 'glucose', 'pressure', 'triceps',
        'insulin','mass', 'pedigree', 'age']]
y = df['diabetes']


def task1(X, y):
    print('\nTask 1\n')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=50)
    
    cvs = []
    loos = []
    non = []
    k = 150
    
    for i in range(k):
        knn = KNeighborsClassifier(n_neighbors=i+1)
        knn.fit(X_train, y_train)
        cv = cross_val_score(knn, X_test, y_test, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        loo = cross_val_score(knn, X_test, y_test, cv=LeaveOneOut(), scoring='neg_mean_absolute_error', n_jobs=-1)
        cvs.append(np.mean(np.abs(cv)))
        loos.append(np.mean(np.abs(loo)))
        y_pred = list(knn.predict(X_test))
        sm = 0
        for j in range(len(y_pred)):
            sm += np.abs(list(y_test)[j] - y_pred[j])
        non.append(sm/j)
    
    x = np.arange(1, k+1, 1, dtype=np.int64)
    plt.figure(figsize=(8,6))
    plt.plot(x, cvs)
    plt.plot(x, loos)   
    plt.plot(x, non) 
    plt.legend(['5-fold CV', 'LOOCV', 'k-NN'])
    plt.xlabel('k')
    plt.ylabel('Mean Absolute Error')
    plt.xticks(ticks=plt.xticks()[0], labels=plt.xticks()[0].astype(int))
    plt.xlim(0, k+1)
    plt.title('Test error of k-NN, with 5-fold CV, LOOCV and without for all k')
    plt.show()

#---TASK 2---

def task2(X, y):
    print('\nTask 2\n')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=50)
    
    gam = LogisticGAM(n_splines=8).gridsearch(X_train.values, y_train)
    
    print(gam.accuracy(X_test, y_test))
    
    fig, axs = plt.subplots(1, 8)
    titles = df.columns[0:8]
    for i, ax in enumerate(axs):
        XX = gam.generate_X_grid(term=i)
        pdep, confi = gam.partial_dependence(term=i, width=.95)
    
        ax.plot(XX[:, i], pdep)
        ax.plot(XX[:, i], confi, c='r', ls='--')
        ax.set_title(titles[i])
    
    
    plt.show()

#---TASK 3---

def task3(X, y):
    print('\nTask 3\n')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=50)
    
    tree = DecisionTreeClassifier().fit(X_train, y_train)
    print('Decision Tree:\n')
    print('Train:', tree.score(X_train, y_train))
    print('Test:', tree.score(X_test, y_test))
    
    probaBag = VotingClassifier([('b',BaggingClassifier())], voting='soft').fit(X_train,y_train)
    consBag = VotingClassifier([('b',BaggingClassifier())], voting='hard').fit(X_train,y_train)
    print('\nBagging with voting:\n')
    print('Probability:')
    print('Train:',probaBag.score(X_train, y_train))
    print('Test:',probaBag.score(X_test, y_test))
    print('\nConsensus:')
    print('Train:',consBag.score(X_train, y_train))
    print('Test:',consBag.score(X_test, y_test))
    
    forest = RandomForestClassifier().fit(X_train, y_train)
    print('\nRandom Forest:\n')
    print('Train:',forest.score(X_train, y_train))
    print('Test:',forest.score(X_test, y_test))
    
    ada = AdaBoostClassifier().fit(X_train, y_train)
    print('\nADABoost:\n')
    print('Train:',ada.score(X_train, y_train))
    print('Test:',ada.score(X_test, y_test))




#---TASK 5---

df2 = pd.read_csv('PimaIndiansDiabetes2.csv', sep=';', index_col=False)
tlist = []
for i in df2['mass']:
    if str(i) == 'nan':
        tlist.append(float('nan'))
    else:
        i = float(i.replace(',', '.'))
        tlist.append(i)
        
df2['mass'] = tlist

tlist = []
for i in df2['pedigree']:
    if str(i) == 'nan':
        tlist.append(float('nan'))
    else:
        i = float(i.replace(',', '.'))
        tlist.append(i)
df2['pedigree'] = tlist


tlist = []
for i in df2['diabetes']:
    if i == 'neg':
        i = 0
    else:
        i = 1
    tlist.append(i)

df2['diabetes'] = tlist
df2 = df2.iloc[: , 1:]

df2 = df2.dropna()


X2 = df2[['pregnant', 'glucose', 'pressure', 'triceps',
        'insulin','mass', 'pedigree', 'age']]
y2 = df2['diabetes']

def taskAll(X, y):
    task1(X,y)
    task2(X,y)
    task3(X,y)


    
taskAll(X, y)
taskAll(X2, y2)

