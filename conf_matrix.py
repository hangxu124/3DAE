import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve

y_true = np.load("buffer_t.npy")
y_pred = np.load("buffer_p.npy")
print (y_pred)
np.savetxt("pred.txt",y_pred)

cm = confusion_matrix(y_true, y_pred)
print (cm)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
print (cm_normalized)


y_pred = (y_pred + 1).astype(int)
with open("pred.txt","r") as pd:
	predict = pd.readlines()

with open("annotation_Dad/vallist01.txt","r") as val:
	v = val.readlines()[:3140]

for a,b in zip(predict,v):
	with open("result_front_wo.txt","a") as result:
		result.write(b[:-1]+ " "+a )