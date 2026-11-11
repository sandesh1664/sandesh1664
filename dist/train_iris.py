"""Reproduce the small browser classifier. Educational benchmark, not real-world validation."""
import json
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
x,y=load_iris(return_X_y=True)
a,b,c,d=train_test_split(x,y,test_size=.2,random_state=42,stratify=y)
m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=42)).fit(a,c)
s=m.named_steps['standardscaler'];l=m.named_steps['logisticregression'];pred=m.predict(b)
result={'classes':['setosa','versicolor','virginica'],'features':['Sepal length','Sepal width','Petal length','Petal width'],'mean':s.mean_.tolist(),'scale':s.scale_.tolist(),'coef':l.coef_.tolist(),'intercept':l.intercept_.tolist(),'accuracy':float(accuracy_score(d,pred)),'confusion':confusion_matrix(d,pred).tolist(),'train_count':len(a),'test_count':len(b),'seed':42,'examples':[[5.1,3.5,1.4,.2],[6.,2.9,4.5,1.5],[6.5,3.,5.8,2.2]]}
Path('dist/data/iris-model.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'train':len(a),'test':len(b),'accuracy':result['accuracy'],'confusion':result['confusion']}))
