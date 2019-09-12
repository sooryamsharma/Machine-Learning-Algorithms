import numpy as np

class LinearRegression:
    def __init__(self, training_file, degrees, lmda):
        self.training_file = training_file
        self.degrees = degrees
        self.lmda = lmda
        
    def load_data(self):
        # reading training data file
        training_file = open(self.training_file, "r")
        lines_training = training_file.readlines()
        # geting number of dimensions
        self.dimensions = len(list(filter(None, lines_training[1].split(" "))))  
        # declaring empty array to stack the data
        self.train_data = np.zeros((1,self.dimensions))
        for line in lines_training:
            x = line.split(" ")
            x = list(filter(None, x))   # getting rid of empty values
            x = list(map(float, x))    # keeping the relative order between the data values
            self.train_data = np.vstack((self.train_data, x))
        self.train_data = self.train_data[1:]
     
    def get_phi(self):
        no_of_attr = self.train_data.shape[0]
        degrees = self.degrees
        self.phi = np.ones((no_of_attr, degrees+1))   # design matrix: N x M 
        for i in range(no_of_attr):
            for j in range(degrees):
                deg = j + 1
                self.phi[i,j+1] = self.train_data[i,0]**deg
    
    def get_weights(self):
        if(self.lmda==0):
            self.weights = np.dot(np.dot(np.linalg.inv(np.dot(self.phi.T, self.phi)), self.phi.T), self.train_data[:,1])
        else:
            self.weights = np.dot(np.dot(np.linalg.inv(np.dot(self.lmda, np.identity(self.degrees+1)) + np.dot(self.phi.T, self.phi)), self.phi.T), self.train_data[:,1])
        return self.weights

def main():
    
    ''' User input should follow the given format:
        -----------------------------------------------------------------------------------------
        linear_regression <training_file> <degree> <λ>
        -----------------------------------------------------------------------------------------
        The arguments provide to the program the following information:
        > The first argument is the path name of the training file
        > The second argument is a number. This number should be either 1 or 2. If the number is 1, 
          program should fit a line to the data. If the number is 2, program should fit a second-degree 
          polynomial to the data.
        > The third number is a non-negative real number (it can be zero or greater than zero). 
          This is the value of λ that you should use for regularization. If λ = 0, then no regularization is used.          

        # Sample inputs -->
        linear_regression sample_data1.txt 1 0
        linear_regression sample_data1.txt 2 0
        linear_regression sample_data1.txt 2 0.001
        linear_regression sample_data1.txt 2 1
    '''

    print("\nPlease provide the following: training file's path, degree and lambda:\n",
	"linear_regression <training_file> <degree> <λ>\n",
	"Please enter here:")
    user_input = input()   # user input in the aforementioned fashion
    input_list = user_input.split()   # converting the input into a list
    
    lr_model = LinearRegression(training_file=input_list[1], degrees=int(input_list[2]), lmda=float(input_list[3]))
    lr_model.load_data()
    lr_model.get_phi()
    weights = lr_model.get_weights()
    for i in range(weights.shape[0]):
        print("w{:d} = {:.4f}".format(i,weights[i]))
    if int(input_list[2])==1:
        print("w2 = {:.4f}\n".format(0))

    
main()



