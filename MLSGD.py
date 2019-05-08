from mnist import MNIST
from sklearn.linear_model import SGDClassifier
import numpy as np
import time
import matplotlib.pyplot as plt

start_time = time.time()

mndata = MNIST('samples')

images, labels = mndata.load_training()

new_images = []
new_labels = []
average_array = []

batch_size = [12000, 6000, 3000, 2000, 1000, 600, 300, 200, 150,
              120, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
x_range2 = []
for j in batch_size:
    for i in range(len(images)):
        if i % j == 0:
            new_images.append(images[i])
            new_labels.append(labels[i])
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=400, tol=1e-3, learning_rate='constant', eta0=.1)
    x_range2.append(len(new_images))
    clf.fit(new_images, new_labels)
    new_labels.clear()
    new_images.clear()

    training_time = time.time()

    test_images, test_labels = mndata.load_testing()

    # start_predict_time = time.time()
    guesses = clf.predict(test_images)
    # predict_time = time.time()
    array_sum = np.sum(guesses == test_labels)
    average = array_sum/(len(guesses))
    average_array.append(average)
    # print("training time: ", training_time - start_time)
    # print("predict time: ", predict_time - start_predict_time)
plt.plot(x_range2, average_array)

plt.xlabel("Batch Size")
plt.ylabel("Prediction Accuracy")
plt.title("Batch Size vs. Prediction Accuracy (learning rate = 0.1)")
plt.show()
print(average_array)
print(x_range2)

new_labels.clear()
new_images.clear()
average_array.clear()


# Varying learning rate

 # learning_rate = [.1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1]
  #learning_rate = [.1, .09, .08, .07, .06, .05, .04, .03, .02, .01, .001]
learning_rate = [.001, .01, .02, .03, .04, .05, .06, .07, .08, .09, .1]
for j in learning_rate:
     for i in range(len(images)):
         if i % 10 == 0:
             new_images.append(images[i])
             new_labels.append(labels[i])
     clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=400, tol=1e-3, learning_rate='adaptive', eta0=j, n_jobs=-1)
     # x_range2.append(len(new_images))
     clf.fit(new_images, new_labels)

     test_images, test_labels = mndata.load_testing()

     guesses = clf.predict(test_images)
     array_sum = np.sum(guesses == test_labels)
     average = array_sum/(len(guesses))
     average_array.append(average)
plt.plot(learning_rate, average_array)

plt.xlabel("Learning Rate")
plt.ylabel("Prediction Accuracy")
plt.title("Learning Rate vs. Prediction Accuracy (n=6000)")
plt.show()
print(average_array)
print(learning_rate)
