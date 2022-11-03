import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression, Lasso, LassoLarsIC, Ridge
from sklearn.ensemble import BaggingRegressor
from sklearn.feature_selection import SequentialFeatureSelector
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from statsmodels.gam.api import GLMGam, BSplines
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#---TASK 1---
df = pd.read_csv('qsar_aquatic_toxicity.csv', sep=';', names=['TPSA', 'SAacc', 'H050', 'ML0GP', 'RDCHI',
                                                                'GATS1p', 'nN', 'C040', 'LC50'])
ddf = df[['H050','nN', 'C040']]

full = []
for i in ddf:
    temp = []
    for j in ddf[i]:
        if j != 0:
            j = 1
        temp.append(j)
    full.append(temp)

ddf = df.copy()

ddf['H050'] = full[0]
ddf['nN'] = full[1]
ddf['C040'] = full[2]

def task1(df, ddf, ret=False):
    print('\nTask 1\n')
    X_train, X_test, y_train, y_test = train_test_split(df[['TPSA', 'SAacc', 'H050', 'ML0GP', 'RDCHI',
                                                                        'GATS1p', 'nN', 'C040']], df['LC50'],
                                                                test_size=1/3, random_state=50)
    
    dX_train, dX_test, dy_train, dy_test = train_test_split(ddf[['TPSA', 'SAacc', 'H050', 'ML0GP', 'RDCHI',
                                                                    'GATS1p', 'nN', 'C040']], ddf['LC50'],
                                                            test_size=1/3, random_state=50)
    
    reg = Pipeline([('scl',StandardScaler()),('lreg',LinearRegression())]).fit(X=X_train, y=y_train)
    dreg = Pipeline([('scl',StandardScaler()),('lreg',LinearRegression())]).fit(X=dX_train, y=dy_train)
    
    print(f'Normal Coefficients: {reg["lreg"].coef_}')
    print(f'Dichotomized Coefficients: {dreg["lreg"].coef_}')
    print(f'Ratio: {reg["lreg"].coef_/dreg["lreg"].coef_}')
    
    pred = reg.predict(X_test)
    dpred = dreg.predict(dX_test)
    
    sm = 0
    dsm = 0
    for j in range(len(pred)):
                sm += np.abs(list(y_test)[j] - pred[j])
                dsm += np.abs(list(dy_test)[j] - dpred[j])
    
    sm = sm/j
    dsm = dsm/j
    
    plt.figure()
    line = np.linspace(0,10,10)
    plt.plot(line,line, color='black')
    plt.scatter(y_test, pred, color='b',alpha=0.75)
    plt.scatter(dy_test, dpred, color='r', alpha=0.75)
    a, b = np.polyfit(y_test, pred, 1)
    c, d = np.polyfit(dy_test, dpred, 1)
    plt.plot(y_test, a*y_test+b, color='b', linestyle=':')
    plt.plot(dy_test, c*dy_test+d, color='r', linestyle=':')
    plt.legend(['Observed', 'Normal', 'Dichotomized'])
    plt.xlabel('Observed LC50')
    plt.ylabel('Predicted LC50')
    plt.show()
    
    print('Normal MAE: %f' %(sm))
    print('Dichotomized MAE: %f' %(dsm))
    
    if ret == True:
        toreturn = []
        toreturn.append(sm)
        toreturn.append(dsm)
        sm = 0
        dsm = 0
        pred = reg.predict(X_train)
        dpred = dreg.predict(dX_train)
        for j in range(len(pred)):
                sm += np.abs(list(y_train)[j] - pred[j])
                dsm += np.abs(list(dy_train)[j] - dpred[j])
        
        sm = sm/j
        dsm = dsm/j
        toreturn.append(sm)
        toreturn.append(dsm)
        
        return toreturn
    
#---TASK 2---
def task2(df, ddf):
    print('\nTask 2\n')
    tests = np.linspace(0.10, 0.5, 200)
    lossList = []
    dlossList = []
    pipe = make_pipeline(StandardScaler(), LinearRegression())
    
    for i in tests:
        X_train, X_test, y_train, y_test = train_test_split(df[['TPSA', 'SAacc', 'H050', 'ML0GP', 'RDCHI',
                                                                    'GATS1p', 'nN', 'C040']], df['LC50'],
                                                            test_size=i, random_state=50)
        
        dX_train, dX_test, dy_train, dy_test = train_test_split(ddf[['TPSA', 'SAacc', 'H050', 'ML0GP', 'RDCHI',
                                                                    'GATS1p', 'nN', 'C040']], ddf['LC50'],
                                                            test_size=i, random_state=50)
        
        reg = pipe.fit(X_train, y_train)
        dreg = pipe.fit(dX_train, dy_train)
        pred = reg.predict(X_test)
        dpred = dreg.predict(dX_test)
        sm = 0
        dsm = 0
        for j in range(len(pred)):
                sm += np.abs(list(y_test)[j] - pred[j])
                dsm += np.abs(list(dy_test)[j] - dpred[j])
    
        sm = sm/j
        dsm = dsm/j
        lossList.append(sm)
        dlossList.append(dsm)
    
    plt.figure()    
    plt.plot(tests, lossList, color='red')
    plt.plot(tests, dlossList, color='dodgerblue')
    plt.plot([0.1, 0.5], [np.average(lossList)]*2, color='red', linestyle='-.')
    plt.plot([0.1, 0.5], [np.average(dlossList)]*2, color='dodgerblue', linestyle='-.')
    plt.legend(['Normal', 'Dichotomized', 'Avg. Normal', 'Avg. Dichotomized'])
    plt.xlabel('Test Size')
    plt.ylabel('Mean Absolute Error')
    plt.title('Mean Absolute Error for different test sizes using Linear Regression')
    plt.show()


#---TASK 3---

def task3(df, ret=False):
    print('\nTask 3\n')
    X = df[['TPSA', 'SAacc', 'H050', 'ML0GP', 'RDCHI', 'GATS1p', 'nN', 'C040']]
    y = df['LC50']
    X_train, X_test, y_train, y_test = train_test_split(df[['TPSA', 'SAacc', 'H050', 'ML0GP', 'RDCHI',
                                                                'GATS1p', 'nN', 'C040']], df['LC50'],
                                                        test_size=1/3, random_state=50)
    feature_names = np.array(X.columns)
    las = make_pipeline(StandardScaler(), Lasso(alpha=0.05))
    lasAIC = make_pipeline(StandardScaler(), LassoLarsIC(criterion='aic')).fit(X_train, y_train)
    lasBIC = make_pipeline(StandardScaler(), LassoLarsIC(criterion='bic')).fit(X_train, y_train)
    las_f = SequentialFeatureSelector(las, direction='forward', cv=5, scoring='neg_mean_absolute_error').fit(X, y)
    las_b = SequentialFeatureSelector(las, direction='backward', cv=5, scoring='neg_mean_absolute_error').fit(X, y)

    print('R2-score with AIC criteria: %f' %(lasAIC.score(X_test, y_test)))
    print('R2-score with BIC criteria: %f' %(lasBIC.score(X_test, y_test)))

    print('Selected features from forward selection are:')
    for i in list(feature_names[las_f.get_support()]):
        print(i)
    
    print('\nSelected features from backward selection are:')
    for i in list(feature_names[las_b.get_support()]):
        print(i)

    XfTrain = X_train[list(feature_names[las_f.get_support()])]
    XfTest = X_test[list(feature_names[las_f.get_support()])]
    XbTrain = X_train[list(feature_names[las_b.get_support()])]
    XbTest = X_test[list(feature_names[las_b.get_support()])]
    
    
    fullF = las.fit(XfTrain, y_train)
    fullB = las.fit(XbTrain, y_train)
    
    print('R2-score with forward selection: %f' %(fullF.score(XfTest, y_test)))
    print('R2-score with backward selection: %f' %(fullB.score(XbTest, y_test)))
    
        
    if ret == True:
        toreturn = []
        AICtestpred = lasAIC.predict(X_test)
        AICtrainpred = lasAIC.predict(X_train)
        BICtestpred = lasBIC.predict(X_test)
        BICtrainpred = lasBIC.predict(X_train)
        aicsm = 0
        bicsm = 0
        for i in range(len(AICtestpred)):
            aicsm += np.abs(list(y_test)[j] - AICtestpred[j])
            bicsm += np.abs(list(y_test)[j] - BICtestpred[j])
        toreturn.append(aicsm/i)
        toreturn.append(bicsm/i)
        aicsm = 0
        bicsm = 0
        for i in range(len(AICtrainpred)):
            aicsm += np.abs(list(y_train)[j] - AICtrainpred[j])
            bicsm += np.abs(list(y_train)[j] - BICtrainpred[j])
        
        toreturn.append(aicsm/i)
        toreturn.append(bicsm/i)
        
        
        forwardPred = fullF.predict(XfTest)
        backwardPred = fullB.predict(XbTest)
        aicsm = 0
        bicsm = 0
        for i in range(len(forwardPred)):
            aicsm += np.abs(list(y_test)[j] - forwardPred[j])
            bicsm += np.abs(list(y_test)[j] - backwardPred[j])
            
        toreturn.append(aicsm/i)
        toreturn.append(bicsm/i)
        
        forwardPred = fullF.predict(XfTrain)
        backwardPred = fullB.predict(XbTrain)
        aicsm = 0
        bicsm = 0
        for i in range(len(forwardPred)):
            aicsm += np.abs(list(y_train)[j] - forwardPred[j])
            bicsm += np.abs(list(y_train)[j] - backwardPred[j])
            
            
        toreturn.append(aicsm/i)
        toreturn.append(bicsm/i)
        
        return toreturn
    
    
#---TASK 4---
def task4(df, ret=False):
    print('\nTask 4\n')
    X = df[['TPSA', 'SAacc', 'H050', 'ML0GP', 'RDCHI', 'GATS1p', 'nN', 'C040']]
    y = df['LC50']
    X_train, X_test, y_train, y_test = train_test_split(df[['TPSA', 'SAacc', 'H050', 'ML0GP', 'RDCHI',
                                                                'GATS1p', 'nN', 'C040']], df['LC50'],
                                                        test_size=1/3, random_state=50)
    alphas = np.linspace(0.1, 12, 300)
    bags = []
    cvs = []
    store = 1000
    for i in alphas:
        pipe = make_pipeline(StandardScaler(), Ridge(alpha = i))
        bag = BaggingRegressor(pipe, bootstrap=True).fit(X_train, y_train)
        bags.append(bag.score(X_test, y_test))
        cv = cross_validate(pipe, X, y, cv=7)
        cvs.append(np.mean(cv['test_score']))
        if np.mean(cv['test_score']) < store:
            bestAlpha = i
            store = i
    
    plt.figure()
    plt.plot(alphas, bags)
    plt.plot(alphas, cvs)
    plt.xlabel('alpha')
    plt.ylabel('Test score')
    plt.legend(['Bootstrap', 'CV'])
    plt.title('Test score for alpha selection with bootstrap and CV respectively')
    plt.text(0, .402, 'Best alpha: %g' %(bestAlpha))
    plt.show()
    
    
    if ret==True:
        pipe = make_pipeline(StandardScaler(), Ridge(alpha = bestAlpha))
        bag = BaggingRegressor(pipe, bootstrap=True).fit(X_train, y_train)
        cv = cross_validate(pipe, X, y, cv=7, return_train_score=True, scoring='neg_mean_absolute_error')
        toreturn = []
        bagtest = bag.predict(X_test)
        bagtrain = bag.predict(X_train)
        sm = 0
        for j in range(len(bagtest)):
            sm += np.abs(list(y_test)[j] - bagtest[j])
        toreturn.append(sm/j)
        
        sm = 0
        for j in range(len(bagtrain)):
            sm += np.abs(list(y_train)[j] - bagtrain[j])
        toreturn.append(sm/j)
        
        toreturn.append(-np.mean(cv['test_score']))
        toreturn.append(-np.mean(cv['train_score']))
        return toreturn

#---TASK 5---

def task5(df, ret=False):
    print('\nTask 5\n')
    splines = [df[['SAacc', 'ML0GP']],df[['nN', 'H050']], df[['GATS1p', 'SAacc']]]
    store = 100
    for i in splines:
        bs = BSplines(i, df=[15,20], degree=[3,3])
        
        gam_bs = GLMGam.from_formula('LC50 ~ RDCHI + TPSA', data=df, smoother=bs)
        
        res_bs = gam_bs.fit()
        
        pred = (res_bs.predict())
        sm = 0
        for j in range(len(pred)):
            sm += np.abs(list(df['LC50'])[j] - pred[j])
        
        sm = sm/j
        print('MAE: %f with %s spline' %(sm, str(list(i.columns))))
        if ret==True:
            if sm < store:
                store = sm
            
            
    if ret==True:
        return [store]
            
    
#---TASK 6---

def task6(df, ret=False):
    print('\nTask 6\n')
    X_train, X_test, y_train, y_test = train_test_split(df[['TPSA', 'SAacc', 'H050', 'ML0GP', 'RDCHI',
                                                                'GATS1p', 'nN', 'C040']], df['LC50'],
                                                        test_size=1/3, random_state=50)
    
    dtree = DecisionTreeRegressor()
    path = dtree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    
    trees = []
    for i in ccp_alphas:
        dtree = DecisionTreeRegressor(random_state=0, ccp_alpha=i)
        dtree.fit(X_train, y_train)
        trees.append(dtree)
    
    
    trees = trees[:-1] #remove case where tree is just one node
    ccp_alphas = ccp_alphas[:-1] #ditto
    node_counts = [dtree.tree_.node_count for dtree in trees]
    depth = [dtree.tree_.max_depth for dtree in trees]
    
    
    test_scores = [dtree.score(X_test, y_test) for dtree in trees]
    bestAlpha = ccp_alphas[test_scores.index(np.max(test_scores))]
    
    
    #selecting best parameters:
    plt.figure(figsize=((12,10)))
    tree.plot_tree(DecisionTreeRegressor(ccp_alpha=bestAlpha).fit(X_train, y_train), rounded=True, feature_names=(X_train.columns))
    
    
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()
    
    if ret==True:
        toreturn = []
        testpred = DecisionTreeRegressor(ccp_alpha=bestAlpha).fit(X_train, y_train).predict(X_test)
        trainpred = DecisionTreeRegressor(ccp_alpha=bestAlpha).fit(X_train, y_train).predict(X_train)
        sm = 0
        for i in range(len(testpred)):
            sm += np.abs(list(y_test)[i] - testpred[i])
        toreturn.append(sm/i)
        sm = 0
        for i in range(len(trainpred)):
            sm += np.abs(list(y_train)[i] - trainpred[i])
    
        toreturn.append(sm/i)
        
        return toreturn
    
def task1to6(df, ddf):
    task1(df, ddf)
    task2(df, ddf)
    task3(df)
    task4(df)
    task5(df)
    task6(df)
    
def task7(df, ddf):
    compare = {} #dict to store all returned values in. The key corresponding to the task contains a list with associated test/train errors as follows:
    compare['1'] = task1(df, ddf, ret=True) #test error, dichotomized test error, train error, dichotomized train error
    compare['3'] = task3(df, ret=True) #AIC test error, BIC test error, AIC train error, BIC train error, forward test error, backward test error,las forward train error, backward train error
    compare['4'] = task4(df, ret=True) #bootstrap test error, bootstrap train error, cv test error, cv train error
    compare['5'] = task5(df, ret=True) #best test error
    compare['6'] = task6(df, ret=True) #tree test error, tree train error
    
    return compare

compareDict = task7(df, ddf)
