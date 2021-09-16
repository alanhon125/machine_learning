from sklearn.datasets import load_iris
import numpy as np
from math import sqrt
from math import exp
from math import pi
from math import log
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class NBC:

    def __init__(self, X, y):
        self.X, self.y = X, y
        self.N, self.D = self.X.shape  # Length of the training set & Dimension of the vector of features
        self.c = list(set(self.y)) # store distinct class of y
        self.Nclass = len(self.c) # No. of classes
        self.Ntrain = int(0.8 * self.N)
        self.shuffler = np.random.permutation(self.N)
        self.Xtrain = self.X[self.shuffler[:self.Ntrain]]
        self.ytrain = self.y[self.shuffler[:self.Ntrain]]
        self.Xtest = self.X[self.shuffler[self.Ntrain:]]
        self.ytest = self.y[self.shuffler[self.Ntrain:]]
        self.normdist_set = {}  # for each class j, store the N(mean, stdev) of feature xi
        self.hot_vec_y = None
        self.prob_set = None # store class probability in training stage by class dictionary
        self.mean_arr = None # store array of means of normal distribution of p(xi|y=ci)
        self.std_arr = None # store array of stdev of normal distribution of p(xi|y=ci)

    # Gaussian probability distribution function for x
    def normal_dist(self, x, mean, stdev):
        stdev = 10^-6 if stdev == 0 else stdev
        exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent

    # encoding classes one hot vectors
    def one_hot_encoding(self,X):
        cat = list(set(X.flatten()))
        cat.sort()
        cat_vectors = [[] for _ in range(self.Nclass)]
        for l in range(len(X)):
            for c in range(self.Nclass):
                cat_vectors[c].append(1) if X[l] == cat[c] else cat_vectors[c].append(0)
        return np.transpose(cat_vectors)

    # Calculate the class probability πc = p(y=ci)
    def class_prob(self):
        self.hot_vec_y = self.one_hot_encoding(self.ytrain)
        self.prob_set = np.sum(self.hot_vec_y,axis=0)/self.Ntrain

    def fit(self):
        dot_prod = np.dot(self.Xtrain.T, self.hot_vec_y)
        dot_prod_size = np.dot(np.sign(self.Xtrain.T), np.sign(self.hot_vec_y))
        self.mean_arr = dot_prod / dot_prod_size

        dist_to_mean = [[(self.Xtrain[i,j] - self.mean_arr[j,self.ytrain[i]])**2 for j in range(self.D)] for i in range(self.Ntrain)]
        dist_to_mean = np.array(dist_to_mean)
        self.std_arr = np.sqrt(np.dot(dist_to_mean.T, self.hot_vec_y)/ (dot_prod_size-np.sign(dot_prod_size)))

        for i in range(self.Nclass): # For each class, compute statistics by feature
            for j in range(self.D):
                print('Given y={}, x_{} ~N(µ={:.2f}, σ={:.2f}, n={:.0f})'.format(i,j+1,self.mean_arr[j,i],self.std_arr[j,i],dot_prod_size[j,i]))

    def predict(self, example):
        class_predict = None  # Final result
        log_max_prob = -10 ^ 6  # initiate log maximum joint probability for comparison
        for cat in self.c:
            log_prob = log(self.prob_set[cat])
            for dim in range(self.D):
                prob = self.normal_dist(example[dim], self.mean_arr[dim,cat], self.std_arr[dim,cat])
                try:
                    log_cond_prob = log(prob)
                    log_prob += log_cond_prob
                except ValueError:
                    log_prob -=10^6
            # if we have a greater joint prob. for this output than the existing maximum, replace with it and store prediction
            if log_prob > log_max_prob:
                log_max_prob = log_prob
                class_predict = cat
        return class_predict

    def confusion_matrix_plot(self, ypred):
        cm = confusion_matrix(self.ytest, ypred, labels=self.c)
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.c)
        cm_disp.plot()
        plt.show()

##################################
if __name__ == "__main__":
    iris = load_iris()
    X, y = iris['data'], iris['target']
    nbc = NBC(X, y)
    nbc.class_prob()
    for i in nbc.c:
        print(f'p(y={i})={nbc.prob_set[i]}')
    nbc.fit()
    total_tests = nbc.N - nbc.Ntrain  # size of validation set
    # Correctly classified examples and incorrectly classified examples
    correct = 0
    ypred = []
    for i in range(total_tests):
        predict = nbc.predict(nbc.Xtest[i])
        correct += (nbc.ytest[i] == predict)
        ypred.append(predict)
    wrong = total_tests - correct
    print('TOTAL EXAMPLES:', total_tests)
    print('CORRECT:', correct)
    print('WRONG:', wrong)
    print('ACCURACY:', correct / total_tests*100, ' %')

    nbc.confusion_matrix_plot(ypred)