import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score

y_true = np.load("buffer_t.npy")
#print (y_true)
y_pred = np.load("buffer_p.npy")
f = 0.5
t = 0.5
prob_front = np.load("prob/prob_sv1_front_depth.npy")
print (np.shape(prob_front))
prob_top   = np.load("prob/prob_sv1_top_depth.npy")
print (np.shape(prob_top))
prob = prob_front[1:,1]*f + prob_top[1:,1] *t
print (np.shape(prob))
pred = np.zeros((6660,1))
for i in range(6660):
	if prob[i,]>0.5:
		pred[i]=1
# np.save("prob_mv2_fusion_depth.npy",prob)
# print (pred)
# np.savetxt("pred.txt",pred)
#np.savetxt("pred.txt",y_pred)
#print (np.shape(y_pred))




cm = confusion_matrix(y_true, pred)
print (cm)
np.set_printoptions(precision=4)
cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
print (cm_normalized)



label = np.loadtxt("vallabel.txt")[:6660]
print (np.shape(label))
auc = roc_auc_score(label,prob)
print (auc)





ind_array = np.arange(2)
x, y = np.meshgrid(ind_array, ind_array)
 
for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if (c > 0.01):
    	plt.text(x_val, y_val, "%0.2f" %(c,), color='red', fontsize=10, va='center', ha='center')
cmap = plt.cm.binary
title='Confusion Matrix'
plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
# xlocations = np.array(range(len(labels)))
# plt.xticks(xlocations, labels, rotation=90)
# plt.yticks(xlocations, labels)
plt.ylabel('True label')
plt.xlabel('Predicted label')


plt.show()

# y_pred = (y_pred + 1).astype(int)
# with open("pred.txt","r") as pd:
# 	predict = pd.readlines()

# with open("annotation_Dad/vallist01.txt","r") as val:
# 	v = val.readlines()[:3140]

# for a,b in zip(predict,v):
# 	with open("result_front_wo.txt","a") as result:
# 		result.write(b[:-1]+ " "+a )