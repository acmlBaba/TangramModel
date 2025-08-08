import numpy as np

#8
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import sys

ou=np.load("./output/out_"+str(sys.argv[1])+'_'+str(sys.argv[2])+".npy", allow_pickle=True)
inp=[[ou[i][j][2] for i in range(6)] for j in range(8)]
out=[[ou[i][j][4] for i in range(6)] for j in range(8)]
inp=sum(inp,[])
out=sum(out,[])

# 既存のaccuracy.npyファイルを読み込みます
if int(sys.argv[2])>1:
    existing_accuracy = np.load('./results/accuracy'+str(sys.argv[1])+'.npy', allow_pickle=True)
    existing_matrix = np.load('./results/confusion_matrix'+str(sys.argv[1])+'.npy', allow_pickle=True)


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
matrix_new = cm

# 既存のaccuracy.npyに新しい値を追加して保存します
if int(sys.argv[2])==1:
    updated_accuracy = accuracy_new
    updated_matrix = [matrix_new]
else:
    updated_accuracy = np.append(existing_accuracy, accuracy_new)
    updated_matrix = np.concatenate((existing_matrix, [matrix_new]), axis=0)


# 更新したaccuracy.npyを保存します
np.save('./results/accuracy'+str(sys.argv[1])+'.npy', updated_accuracy)


np.save('./results/confusion_matrix'+str(sys.argv[1])+'.npy',updated_matrix)
