import matplotlib.pyplot as plt
from custom_dataset import CustomDataset
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import svm
import seaborn as sn


data_path = "C:\\Users\\Mehmet\\Desktop\\yeniANN"

# Initialize the dataset and dataloader
traindataset = CustomDataset(data_path = data_path, train = True, val = False)
trainloader = DataLoader(traindataset, batch_size = len(traindataset), shuffle = True, pin_memory = True, num_workers = 0)
"""
valdataset = CustomDataset(data_path = data_path, train = False, val = True)
valloader = DataLoader(traindataset, batch_size = len(valdataset), shuffle = False, pin_memory = True, num_workers = 0)
"""
testdataset = CustomDataset(data_path = data_path, train = False, val = False)
testloader = DataLoader(testdataset, batch_size = len(testdataset), shuffle = True, pin_memory = True, num_workers = 0)

print('Processing train data')
for data in trainloader:
    train_data = data[0].numpy().reshape((12803, 519*1))  # reshape data for fitting it to kn.fit
    train_label = data[1].numpy()

print('Processing test data')
for data in testloader:
    test_data = data[0].numpy().reshape((1607, 519*1))
    test_label = data[1].numpy()

LABELS = ["BS", "EB", "H", "LB", "LM", "SLP", "SM", "TS", "MV", "YLCV"]
"""
kn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
kn.fit(train_data, train_label)
knn_acc_score = accuracy_score(test_label, kn.predict(test_data)) * 100
print("KNN Accuracy: ({:.5f}%)".format(knn_acc_score))

# Confusion matrix
conf_mat_knn = confusion_matrix(test_label, kn.predict(test_data), normalize = 'true')

fig, ax = plt.subplots(figsize = (20, 10))
sn.heatmap(conf_mat_knn, annot = True, xticklabels = LABELS, yticklabels = LABELS, cmap = "YlGnBu")

ax.set_title("KNN Confusion Matrix ({:.5f}%)".format(knn_acc_score))
plt.savefig("KNN_Confusion_Matrix.png", dpi = 100)
plt.show()
print(conf_mat_knn)
"""
"""
rf=RandomForestClassifier(random_state = 0, n_estimators = 100)
rf.fit(train_data, train_label)
rf_acc_score = accuracy_score(test_label, rf.predict(test_data)) * 100
print("Random Forest Accuracy: ({:.5f}%)".format(rf_acc_score))

# Confusion matrix
conf_mat_RF = confusion_matrix(test_label, rf.predict(test_data), normalize = 'true')

fig, ax = plt.subplots(figsize = (20, 10))
sn.heatmap(conf_mat_RF, annot = True, xticklabels = LABELS, yticklabels = LABELS, cmap = "YlGnBu")

ax.set_title("Random Forest Confusion Matrix ({:.5f}%)".format(rf_acc_score))
plt.savefig("RF_Confusion_Matrix.png", dpi = 100)
plt.show()
print(conf_mat_RF)
"""
"""
svm_clf = svm.SVC(decision_function_shape = 'ovo')
svm_clf.fit(train_data, train_label)
svm_acc_score = accuracy_score(test_label, svm_clf.predict(test_data)) * 100
print("Multi Class SVM Accuracy: ({:.5f}%)".format(svm_acc_score))

# Confusion matrix
conf_mat_SVM = confusion_matrix(test_label, svm_clf.predict(test_data), normalize = 'true')

fig, ax = plt.subplots(figsize = (20, 10))
sn.heatmap(conf_mat_SVM, annot = True, xticklabels = LABELS, yticklabels = LABELS, cmap = "YlGnBu")

ax.set_title("Multi Class SVM Confusion Matrix ({:.5f}%)".format(svm_acc_score))
plt.savefig("SVM_Confusion_Matrix.png", dpi = 100)
plt.show()
print(conf_mat_SVM)
"""

tree_clf = tree.DecisionTreeClassifier()
tree_clf.fit(train_data, train_label)
tree_acc_score = accuracy_score(test_label, tree_clf.predict(test_data)) * 100
print("Decision Tree Accuracy: ({:.5f}%)".format(tree_acc_score))

# Confusion matrix
conf_mat_tree = confusion_matrix(test_label, tree_clf.predict(test_data), normalize = 'true')

fig, ax = plt.subplots(figsize = (20, 10))
sn.heatmap(conf_mat_tree, annot = True, xticklabels = LABELS, yticklabels = LABELS, cmap = "YlGnBu")

ax.set_title("Decision Tree Confusion Matrix ({:.5f}%)".format(tree_acc_score))
plt.savefig("Decision_Tree_Confusion_Matrix.png", dpi = 100)
plt.show()
print(conf_mat_tree)
