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

        self.train_data = np.zeros((1,self.dimensions))   # declaring empty array to stack the data
        self.test_data = np.zeros((1,self.dimensions))   # declaring empty array to stack the data
        self.labels = np.zeros((1,1))   # declaring empty array for the labels
        
        for line in lines_training:
            x = line.split(" ")
            x = list(filter(None, x))   # getting rid of empty values
            x = list(map(float, x))    # keeping the relative order between the data values
            self.train_data = np.vstack((self.train_data, x))

        for line in lines_test:
            x = line.split(" ")
            x = list(filter(None, x))   # getting rid of empty values
            x = list(map(float, x))    # keeping the relative order between the data values
            self.test_data = np.vstack((self.test_data, x))

        self.train_data = self.train_data[1:]
        self.test_data = self.test_data[1:]
        
class PCA(DataHandler):
    def __init__(self, training_file, test_file, M, iterr):
        DataHandler.__init__(self, training_file, test_file)
        self.M = M
        self.iterr = iterr

    def get_eigenVectors(self):
        X = self.train_data[:,:-1]
        U = np.zeros((X.shape[1], self.M))
        for d in range(self.M):
            # calculating covariance matrix
            S = np.cov(X.T)
            # computing U using power method
            U[:,d] = self.power_method(A=S, D=X.shape[1])
            # calculating projections
            for n in range(X.shape[0]):
                X[n,:] = X[n,:] - np.dot(U[:,d], np.dot(X[n,:],U[:,d]).T).T
        
        '''Printing Eigen Vectors'''
        for d in range(self.M):
            print("\nEigenvector {:d}\n".format(d))
            for n in range(X.shape[1]):
                print("{:d} : {:.4f}".format(n+1, U[n, d]))
        
        '''Calculating and printing projections of the test object'''
        X_test = self.test_data[:,:-1].T
        proj_Xtest = np.dot(U.T, X_test)
        
        for n in range(X_test.shape[0]):
            print("\nTest object {:d}".format(n))
            for d in range(self.M):
                print("{:d} : {:.4f}".format(d, proj_Xtest[d,n]))
            
    def power_method(self, A, D):
        b = np.random.rand(D,1)
        for k in range(self.iterr):
            b = np.dot(A,b)
            b = np.divide(b, np.linalg.norm(b,2))
        return b.T

def main():
    
    ''' User input should follow the given format:
        ---------------------------------------------------------------------------------------------
        pca_power <training_file> <test_file> <M> <iterations>
        ---------------------------------------------------------------------------------------------
        The arguments provide to the program the following information:
        > The first argument, <training_file>, is the path name of the training file, where the training data is stored. 
          The path name can specify any file stored on the local computer.
        > The second argument, <test_file>, is the path name of the test file, where the test data is stored. 
          The path name can specify any file stored on the local computer.
        > The third argument, <M>, specifies the dimension of the output space of the projection. In other words, you will use 
          the <M> with the largest eigenvalues to define the projection matrix.
        > The fourth argument, <iterations>, is a number greater than or equal to 1, that specifies the number of iterations 
          for the power method. Slide 44 in the slides on PCA describes how to use the power method to find the top eigenvector, 
          using a sequence bk. You should stop calculating this sequence after the specified number of iterations, and use the 
          last bk (where k=<iterations>) as the eigenvector.

        # Sample inputs -->
        pca_power pendigits_training.txt pendigits_test.txt 1 10
        pca_power satellite_training.txt satellite_test.txt 2 20
        pca_power yeast_training.txt yeast_test.txt 3 30
    '''
    
    print("\nPlease provide the following: training data path, test data path, output dimension - M, iterations:\n",
    "pca_power <training_file> <test_file> <M> <iterations>\n",
    "Please enter here:")
    user_input = input()   # user input in the aforementioned fashion
    input_list = user_input.split()   # converting the input into a list

    pca = PCA(training_file=input_list[1], test_file=input_list[2], M=int(input_list[3]), iterr=int(input_list[4]))
    pca.load_data()
    pca.get_eigenVectors()

main()

