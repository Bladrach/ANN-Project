from train import *
import glob
import matplotlib.ticker as ticker
from matplotlib import cm
from sklearn.metrics import confusion_matrix
import seaborn as sn


directory = "./saved_models_pv"
files = list(filter(os.path.isfile, glob.glob(directory + "/*.pth")))
files.sort(key=lambda x: os.path.getmtime(x))
acc_list = []
epoch_list = []

pred_list = []
label_list = []

for filename in files:
    net.load_state_dict(torch.load(filename))
    correct = 0
    total = 0
    
    print(filename)
    with torch.no_grad():
        since = time.time()
        for data in testloader:
            inputs = data[0].view(-1, 519)
            #print(inputs.shape)
            inputs = inputs.to(device)
            label = data[1].to(device)

            output = net(inputs)

            _, predicted = torch.max(output.data, 1)
            if (int(filename.split("_")[-1].split(".")[0]) == 10):
                pred_list.append(predicted.item())
                label_list.append(label.item())
            total += label.size(0)
            correct += (predicted == label).sum().item()

    acc_list.append(100 * (correct/total))
    epoch_list.append(int(filename.split("_")[-1].split(".")[0]))
time_elapsed = time.time() - since

fig = plt.figure()
plt.plot(epoch_list, acc_list)
plt.title("Test Accuracy ({:.5f}%)".format(float(acc_list[-1])))
plt.grid()
plt.xlabel("Epoch")
plt.xticks(range(0, 11, 1))
plt.ylabel("Accuracy (%)")
fig.savefig("TestHistory.jpg")
plt.show()

# Confusion matrix
conf_mat = confusion_matrix(label_list, pred_list, normalize = 'true')

LABELS = ["BS", "EB", "H", "LB", "LM", "SLP", "SM", "TS", "MV", "YLCV"]
fig, ax = plt.subplots(figsize = (20, 10))
sn.heatmap(conf_mat, annot = True, xticklabels = LABELS, yticklabels = LABELS, cmap = "YlGnBu")

ax.set_title("Confusion Matrix")
plt.savefig("Confusion_Matrix.png", dpi = 100)
plt.show()
print(conf_mat)
