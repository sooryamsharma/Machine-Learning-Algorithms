import numpy as np

class DataHandler:
    def __init__(self, training_file, test_file):
        self.training_file = training_file
        self.test_file = test_file
        self.train_data = []
        self.test_data = []
        self.dimensions = 0
    
    def load_data(self):
        # reading training data file
        training_file = open(self.training_file, "r")
        lines_training = training_file.readlines()   
        # reading test data file
        test_file = open(self.test_file, "r")
        lines_test = test_file.readlines()  

        self.dimensions = len(list(filter(None, lines_training[1].split(" "))))  # geting number of dimensions

        self.train_data = np.zeros((1,self.dimensions-1))   # declaring empty array to stack the data
        self.test_data = np.zeros((1,self.dimensions-1))   # declaring empty array to stack the data
        self.train_labels = np.zeros((1,1))   # declaring empty array for the labels
        self.test_labels = np.zeros((1,1))   
        
        for line in lines_training:
            x = line.split(" ")
            x = list(filter(None, x))   # getting rid of empty values
            x = list(map(float, x))    # keeping the relative order between the data values
            self.train_data = np.vstack((self.train_data, x[:-1]))
            self.train_labels = np.vstack((self.train_labels, x[-1]))

        for line in lines_test:
            x = line.split(" ")
            x = list(filter(None, x))   # getting rid of empty values
            x = list(map(float, x))    # keeping the relative order between the data values
            self.test_data = np.vstack((self.test_data, x[:-1]))
            self.test_labels = np.vstack((self.test_labels, x[-1]))

        self.train_data = self.train_data[1:]
        self.test_data = self.test_data[1:]  
        self.train_labels = self.train_labels[1:]
        self.test_labels = self.test_labels[1:] 
        
class KNN(DataHandler):
    def __init__(self, training_file, test_file, k):
        DataHandler.__init__(self, training_file, test_file)
        self.k = k
        self.count = 0

    def normalize(self):
        mean = np.mean(self.train_data, axis=0)
        std = np.std(self.train_data, axis=0)    
        for i in range(self.train_data.shape[1]):
            self.train_data[:,i] = (self.train_data[:,i] - mean[i]) / std[i]
            self.test_data[:,i] = (self.test_data[:,i] - mean[i]) / std[i]
    
    def classify(self):
        for i in range(self.test_data.shape[0]):
            distance = self.get_distance(self.test_data[i,:], self.train_data)
            #sorted_distance = distance[distance[:].argsort()]
            sorted_labels = self.train_labels[distance[:].argsort()]
            self.print_results(sorted_labels[:self.k], i)
            
    def get_distance(self, v1, v2):
        v1 = v1.reshape(1,v1.shape[0])
        L2 = v2[:,] - v1
        L2 = np.square(L2)
        L2 = np.sum(L2, axis = 1)
        L2 = np.sqrt(L2)
        return L2
    
    def print_results(self, y_pred, object_id):
        true_class = self.test_labels[object_id]
        (values, counts) = np.unique(y_pred, return_counts=True)
        predicted_class = values[np.argmax(counts)]
        accuracy = 0
        if (true_class[0] == int(predicted_class)):
            accuracy = 1
            self.count += 1
        else:
            accuracy = 0
        print("ID={:5d}, predicted={:3d}, true={:3d}, accuracy={:4.2f}".format(object_id, int(predicted_class), int(true_class[0]), accuracy))
        
    def get_accuracy(self):
        classification_accuracy = self.count/self.test_labels.shape[0]
        print("\nclassification accuracy={:6.4f}".format(classification_accuracy))

def main():
    
    ''' User input should follow the given format:
        ---------------------------------------------------------------------------------------------
        knn_classify <training_file> <test_file> <k>
        ---------------------------------------------------------------------------------------------
        The arguments provide to the program the following information:
        > The first argument, <training_file>, is the path name of the training file, where the training data is stored. 
          The path name can specify any file stored on the local computer. 
        > The second argument, <test_file>, is the path name of the test file, where the test data is stored. The path name 
          can specify any file stored on the local computer.
        > The third argument specifies the value of k for the k-nearest neighbor classifier.
        # Sample inputs -->
        knn_classify pendigits_training.txt pendigits_test.txt 1
        knn_classify pendigits_training.txt pendigits_test.txt 3
        knn_classify pendigits_training.txt pendigits_test.txt 5
    '''
    
    print("\nPlease provide the following: training file path, test file path, value of k:\n",
    "knn_classify <training_file> <test_file> <k>\n",
    "Please enter here:")
    user_input = input()   # user input in the aforementioned fashion
    input_list = user_input.split()   # converting the input into a list
    
    knn = KNN(training_file=input_list[1], test_file=input_list[2], k=int(input_list[3]))
    knn.load_data()
    knn.normalize()
    knn.classify()
    knn.get_accuracy()

main()
