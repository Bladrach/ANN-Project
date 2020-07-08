import numpy as np
import matplotlib.pyplot as plt


methods = ('DT', 'KNN', 'SVM', 'RF', 'MLP')
y_pos = np.arange(len(methods))
accuracy = [84.19, 86.80, 90.79, 95.58, 96.07]  # These accuracies were calculated and saved before with machine_learning_algos.py 
                                                # and train.py scripts. Thus, these numbers were taken by results. It was just easy and convenient to use it like that.  

plt.bar(y_pos, accuracy, align = 'center', alpha = 0.6, color = ['red', 'purple', 'blue', 'cyan', 'green'])

for i, v in enumerate(accuracy):
    plt.text(i - 0.20, 
              v/accuracy[i], 
              accuracy[i],
              fontsize = 12)

plt.xticks(y_pos, methods)
plt.yticks(range(0, 101, 10))
plt.ylabel('Accuracy Percentage')
plt.title('Comparison of Different Algorithms')
plt.savefig("Comparison_of_Different_Algorithms.png")
plt.show()
