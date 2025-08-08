import numpy as np

#8
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import sys

ou=np.load("./output/out_final.npy", allow_pickle=True)
inp=[[ou[i][j][2] for i in range(6)] for j in range(8)]
out=[[ou[i][j][4] for i in range(6)] for j in range(8)]
inp=sum(inp,[])
out=sum(out,[])

print(inp)
print(out)
cm = confusion_matrix(inp, out)
print(classification_report(inp, out))
print(cm)
sns.heatmap(cm, annot=True)
print (accuracy_score(inp, out))
print (f1_score(inp, out,average="micro"))
# 新しいclassification_reportを計算します
report_new = classification_report(inp, out, output_dict=True)

# 新しい正確性（accuracy）を取得
accuracy_new = report_new['accuracy']

np.save('./results/confusion_matrix_final.npy',cm)
