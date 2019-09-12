import numpy as np

class DataHandler:
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = []
    
    def load_data(self):
        # reading training data file
        data_file = open(self.data_file, "r")
        lines = data_file.readlines()   

        n = len(list(filter(None, lines[1].split(" "))))  # geting number of dimensions

        self.data = np.zeros((1,n))   # declaring empty array to stack the data
        
        for line in lines:
            x = line.split(" ")
            x = list(filter(None, x))   # getting rid of empty values
            x = list(map(float, x))    # keeping the relative order between the data values
            self.data = np.vstack((self.data, x))

        self.data = self.data[1:]
        
class SVD(DataHandler):
    def __init__(self, data_file, M, iterr):
        DataHandler.__init__(self, data_file)
        self.M = M
        self.iterr = iterr
        
    def get_eigenVectors(self):
        U = np.zeros((self.data.shape[0], self.M))   # 7 x M matrix
        V = np.zeros((self.data.shape[1], self.M))   # 9 x M matrix
        S = np.identity(self.M)                      # 2 x 2 matrix
        A = self.data

        for d in range(self.M):
            # computing U, S and V using power method
            U[:,d], S[d,d] = self.power_method(A=np.dot(A,A.T))  # A(A.T)
            V[:,d], t = self.power_method(A=np.dot(A.T,A))       # (A.T)A  
            #A = A - np.multiply(np.multiply(U[:,d:d+1], V[:,d:d+1].T), S[d,d])
            A = A - U[:,d:d+1] * V[:,d:d+1].T * S[d,d]
            
        '''Reconstruction matrix'''
        reconstruction_mat = np.dot(U, np.dot(S,V.T))
        
        '''printing U, S, V and reconstruction of the original matrix'''
        self.print_matrix(U, "U")
        self.print_matrix(S, "S")
        self.print_matrix(V, "V")
        self.print_matrix(reconstruction_mat, "Reconstruction (U*S*V')")
    
    def power_method(self, A):
        b = np.ones((A.shape[1]))
        for k in range(self.iterr):
            b = np.dot(A,b)
            lmda = np.sqrt(np.linalg.norm(b,2))
            b = np.divide(b, np.linalg.norm(b,2))
        return b, lmda

    def print_matrix(self, M, matrix_name):
        print("\n\nMatrix {:s}:".format(matrix_name), end ="")
        for row in range(M.shape[0]):
            print("\nRow {:d}:".format(row+1), end ="")
            for col in range(M.shape[1]):
                print("{:8.4f}".format(M[row,col]), end ="")

def main():
    
    ''' User input should follow the given format:
        ---------------------------------------------------------------------------------------------
        svd_power <data_file> <M> <iterations>
        ---------------------------------------------------------------------------------------------
        The arguments provide to the program the following information:
        > The first argument, <data_file>, is the path name of the file where the input matrix is stored. The path name can 
          specify any file stored on the local computer. The data file will have as many lines as the rows of the input matrix. 
          Line n will contain the values in the n-th row of the matrix. Within that line n, values will be separated by white space. 
        > The second argument, <M>, specifies the number of dimensions for the SVD output. In other words, the U matrix 
          should have <M> columns, the V matrix should have <M> columns, and the S matrix should have <M> rows and <M> columns. 
          Remember, the diagonal entries Sd, d of S should contain values that decrease as d increases.
        > The third argument, <iterations>, is a number greater than or equal to 1, that specifies the number of iterations for the 
          power method. Slide 44 in the slides on PCA describes how to use the power method to find the top eigenvector, using a sequence 
          bk. You should stop calculating this sequence after the specified number of iterations, and use the last bk (where k=<iterations>) 
          as the eigenvector.
        # Sample inputs -->
        svd_power input1.txt 2 10
        svd_power input1.txt 4 100
    '''

    print("\nPlease provide the following: training data-file path, output dimension - M, iterations:\n",
    "svd_power <data_file> <M> <iterations>\n",
    "Please enter here:")
    user_input = input()   # user input in the aforementioned fashion
    input_list = user_input.split()   # converting the input into a list

    svd = SVD(data_file=input_list[1], M=int(input_list[2]), iterr=int(input_list[3]))
    svd.load_data()
    svd.get_eigenVectors()
    
main()
