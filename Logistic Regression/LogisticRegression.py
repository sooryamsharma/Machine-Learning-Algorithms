import numpy as np

class LogisticRegression:
    def __init__(self, training_file, degrees, test_file):
        self.training_file = training_file
        self.test_file = test_file
        self.train_data = []
        self.test_data = []
        self.labels = []
        self.classes = []
        self.dimensions = 0
        self.degrees = degrees
    
    def load_data(self):
        # reading training data file
        training_file = open(self.training_file, "r")
        lines_training = training_file.readlines()   
        # reading test data file
        test_file = open(self.test_file, "r")
        lines_test = test_file.readlines()  

        self.dimensions = len(list(filter(None, lines_training[1].split(" "))))  # geting number of dimensions

        self.train_data = np.zeros((1,self.dimensions-1))   # declaring empty array to stack the data
        self.test_data = np.zeros((1,self.dimensions))   # declaring empty array to stack the data
        self.labels = np.zeros((1,1))   # declaring empty array for the labels

        for line in lines_training:
            x = line.split(" ")
            x = list(filter(None, x))   # getting rid of empty values
            x = list(map(float, x))    # keeping the relative order between the data values
            self.train_data = np.vstack((self.train_data, x[:-1]))
            self.labels = np.vstack((self.labels, x[-1]))

        for line in lines_test:
            x = line.split(" ")
            x = list(filter(None, x))   # getting rid of empty values
            x = list(map(float, x))    # keeping the relative order between the data values
            self.test_data = np.vstack((self.test_data, x))

        self.train_data = self.train_data[1:]
        self.test_data = self.test_data[1:]
        self.labels = self.labels[1:]
        self.classes = np.unique(self.labels)
    
    def get_data(self, data):
        if(data=="train"): 
            return self.train_data
        elif(data=="test"):
            return self.test_data
        else:
            print("Please check entered data name!")

    def get_phi(self, data):
        no_of_attr = data.shape[0]
        degrees = self.degrees
        dimensions = self.dimensions - 1
        M = dimensions*degrees+1
        # design matrix / phi matrix: N x M 
        phi = np.ones((no_of_attr, M))   
        for attr in range(no_of_attr):
            m = 1
            for dim in range(dimensions):
                for deg in range(degrees):
                    deg=deg+1
                    phi[attr,m] = data[attr,dim]**deg
                    m += 1
        return phi
    
    def get_weights(self, phi_matrix):
        phi = phi_matrix                    # N x M matrix
        degrees = self.degrees
        dimensions = self.dimensions - 1
        no_of_attr = self.train_data.shape[0]
        M = dimensions*degrees+1
        # weight matrix: (M x 1) initialized with zeros
        weights = np.zeros((M,1))            # M x 1 matrix
        oldE_w = np.zeros((phi.shape[1],1))
        # y: predicted labels for training data
        y = np.zeros((no_of_attr,1))   # N x 1 matrix
        # y_actual: ground truth
        t = self.labels              # N x 1 matrix
        '''
        Converting to Binary Classification Problem
        We have only covered logistic regression for binary classification problems. In this assignment, 
        you should convert the class labels found in the files as follows:
            > If the class label is equal to 1, it stays equal to 1.
            > If the class label is not equal to 1, you must set it equal to 0.
        '''
        t[t!=1]=0
        '''
        Stopping Criteria:
        # Compare the new weight values, computed at this iteration, with the previous weight values. 
          If the sum of absolute values of differences of individual weights is less than 0.001, then 
          you should stop the training.
        # Compute the cross-entropy error, using the new weights computed at this iteration. Compare 
          it with the cross-entropy error computed using the previous value of weights. If the change 
          in the error is less than 0.001, then you should stop the training.
        '''
        stopping_criteria = False
        while stopping_criteria==False:
            for i in range(no_of_attr):
                y[i] = np.dot(weights.T, phi.T[:,i])   # (1 x i) = (1 x M) * (M x i)
                y[i] =  1 / (1 + np.exp(-y[i]))   # sigmoid(z)
            # calculating Error matrix -- E_w
            E_w = np.dot(phi.T, (y - t))   # phi.T * (y - t)
            R_nn = np.zeros((y.shape[0],y.shape[0]))   # (N x N)
            for n in range(y.shape[0]):
                R_nn[n,n] = y[n]*(1-y[n])
            H = np.dot(phi.T, np.dot(R_nn, phi))   # phi.T * R * phi
            new_weights = weights - np.dot(np.linalg.inv(H), E_w)   # inv(phi.T * R * phi) * phi.T * (y - t)
            crossEntropyErr_diff = np.abs(np.sum(E_w) - np.sum(oldE_w))
            stopping_criteria = np.abs(np.sum(new_weights) - np.sum(weights))<0.001 and crossEntropyErr_diff<0.001
            if stopping_criteria==False:
                weights = new_weights
                oldE_w = E_w
        print("\n\n================================Weights================================\n")
        for w in range (weights.shape[0]):
            print("w{:1d} = {:.4f}".format(int(w),float(weights[w])))
        return weights
    
    def get_predictions(self, weights, phi):
        '''
        # predicted class (the result of the classification). If your classification result is a tie, choose one of them randomly.
        # probability of the predicted class given the data. This probability is the output of the classifier if the predicted class is 1. 
          If the predicted class is 0, then the probability is 1 minus the output of the classifier.
        # true class (should be binary, 0 or 1).
        # accuracy. This is defined as follows:
            > If there were no ties in your classification result, and the predicted class is correct, the accuracy is 1.
            > If there were no ties in your classification result, and the predicted class is incorrect, the accuracy is 0.
            > If there were ties in your classification result, and the correct class was one of the classes that tied for best, 
              the accuracy is 1 divided by the number of classes that tied for best.
            > If there were ties in your classification result, since we only have two classes, the accuracy is 0.5.
        '''
        # phi (N x M) matrix
        no_of_attr = self.test_data.shape[0]
        # y_pred: predicted labels for training data
        y_pred = np.zeros((no_of_attr,1))   # N x 1 matrix
        # t: ground truth
        t = self.test_data[:,-1].reshape(no_of_attr,1)
        t[t!=1]=0
        # calculating accuracy
        count=0.0
        accuracy=0.0
        print("\n==============================Predictions==============================\n")
        for i in range(no_of_attr):
            tie=False
            probability=0
            
            y_pred[i] = np.dot(weights.T, phi.T[:,i])   # (1 x i) = (1 x M) * (M x i)
            y_pred[i] =  1 / (1 + np.exp(-y_pred[i]))   # sigmoid(z)
            
            # calculating probability and predicted label
            if (y_pred[i]>0.5):
                probability=y_pred[i]
                y_pred[i]=1
            elif (y_pred[i]==0.5):
                tie=True
                temp = y_pred[i]
                y_pred[i]=np.random.randint(0,1)
                if (y_pred[i]==1):
                    probability=temp
                else:
                    probability=1-temp
            else:
                probability=1-y_pred[i]
                y_pred[i]=0
                
            # calculating accuracy
            if (y_pred[i]==t[i] and tie==False):
                accuracy = 1.0
            elif (y_pred[i]==t[i] and tie==True):
                accuracy = 0.5
            elif (y_pred[i]!=t[i]):
                accuracy = 0.0
            
            count+=accuracy
            # printing result
            print("ID={:5d}, predicted={:3d}, probability = {:.4f}, true={:3d}, accuracy={:4.2f}".format(int(i), 
                                                                                                         int(y_pred[i]), 
                                                                                                         float(probability), 
                                                                                                         int(t[i]), 
                                                                                                         float(accuracy)))
        classification_accuracy = count/no_of_attr
        print("\n================================Accuracy================================\n")
        print("classification accuracy = {:6.4f}\n".format(classification_accuracy))

def main():
    
    ''' User input should follow the given format:
        ---------------------------------------------------------------------------------------------
        logistic_regression <training_file> <degree> <test_file>
        ---------------------------------------------------------------------------------------------
        The arguments provide to the program the following information:
        > The first argument is the path name of the training file
        > The second argument, <degree> is a number equal to either 1 or 2. The degree specifies what 
          function phi you should use. Suppose that you have an input vector x = (x1, x2, ..., xD)T,.
            # If the degree is 1, then phi(x) = (1, x1, x2, ..., xD)T.
            # If the number is 2, then phi(x) = (1, x1, (x1)2, x2, (x2)2..., xD, (xD)2)T.
        > The third argument is the path name of the test file         

        # Sample inputs -->
        logistic_regression pendigits_training.txt 1 pendigits_test.txt
        logistic_regression pendigits_training.txt 2 pendigits_test.txt
    '''
    print("\nPlease provide the following: training file's path, degree and test file's path:\n",
	"logistic_regression <training_file> <degree> <test_file>\n",
	"Please enter here:")
    user_input = input()   # user input in the aforementioned fashion
    input_list = user_input.split()   # converting the input into a list
    
    logr_model = LogisticRegression(training_file=input_list[1], degrees=int(input_list[2]), test_file=input_list[3])
    logr_model.load_data()
    train_data = logr_model.get_data("train")   # get training data
    phi_training = logr_model.get_phi(train_data)   # get phi matrix for training data
    weights = logr_model.get_weights(phi_training)
    test_data = logr_model.get_data("test")
    phi_test = logr_model.get_phi(test_data)
    logr_model.get_predictions(weights, phi_test)

   
    
main()

