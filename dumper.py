import numpy as np
import warnings
warnings.filterwarnings("ignore")
import sklearn.model_selection as ms
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

d = pd.read_csv('winequality-white.csv')
winequalityX = d.iloc[:,:11]
winequalityY = d.iloc[:,11]
Y[0:5]

winequality_trgX, winequality_tstX, winequality_trgY, winequality_tstY = ms.train_test_split(winequalityX, winequalityY, test_size=0.3, random_state=0,stratify=winequalityY)
pipe = Pipeline([('Scale',StandardScaler())])
trgX = pipe.fit_transform(winequality_trgX,winequality_trgY)
trgY = np.atleast_2d(winequality_trgY).T
tstX = pipe.transform(winequality_tstX)
tstY = np.atleast_2d(winequality_tstY).T
trgX, valX, trgY, valY = ms.train_test_split(trgX, trgY, test_size=0.2, random_state=1,stratify=trgY)
tst = pd.DataFrame(np.hstack((tstX,tstY)))
trg = pd.DataFrame(np.hstack((trgX,trgY)))
val = pd.DataFrame(np.hstack((valX,valY)))

tst.to_csv('winequality_test.csv',index=False,header=False)
trg.to_csv('winequality_trg.csv',index=False,header=False)
val.to_csv('winequality_val.csv',index=False,header=False)