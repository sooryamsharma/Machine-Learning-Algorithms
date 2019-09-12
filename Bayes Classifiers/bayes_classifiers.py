import numpy as np
import math

class Naive_Bayes:
    def __init__(self, training_file, test_file):
        self.training_file = training_file
        self.test_file = test_file
        self.train_data = []
        self.test_data = []
        self.labels = []
        self.classes = []
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

    def calc_accuracy(labels, groundTruth, probabs):
        max_probab = np.max(probabs)
        idx = np.nonzero(probabs == max_probab)[0]   # take out the index having probability = max_probab
        predicted_label = 0
        for i in idx:
            i = int(i)
            if labels[i] == groundTruth:
                accuracy = 1/idx.shape[0]
                predicted_label = groundTruth
            else:
                accuracy = 0.0
        predicted_label = int(labels[i])
        return predicted_label, max_probab, accuracy
    
    def print_result(result):
        counter = 0
        #result[0:id, 1:predicted_class, 2:probability, 3:true_class, 4:accuracy]
        for i in range(result.shape[0]):
            print("ID={:5d}, predicted={:3d}, probability= {:.4f}, true={:3d}, accuracy= {:4.2f}".format(int(result[i,0]), 
                                                                                                           int(result[i,1]), 
                                                                                                           result[i,2], 
                                                                                                           int(result[i,3]), 
                                                                                                           result[i,4])) 
            if(result[i,1]==result[i,3]):
                counter+=1
        print("Classification Accuracy= {:.4f}".format(counter/result.shape[0]))

class Histograms(Naive_Bayes):
    def __init__(self, training_file, test_file):
        Naive_Bayes.__init__(self, training_file, test_file)
        
    def train(self, no_of_bins):
        no_of_classes = len(self.classes)
        dimensions = self.dimensions-1
        # initializing required variables with zeros
        self.bin_probability = np.zeros((no_of_classes,dimensions,no_of_bins)) # to store bin probability
        self.bin_min_range = np.zeros((no_of_classes,dimensions,no_of_bins))
        self.bin_max_range = np.zeros((no_of_classes,dimensions,no_of_bins))       
        print("\n============================================ Training Phase =====================================================\n")
        
        for classLabel in self.classes:
            classLabel = int(classLabel)   # float to int
            idx = np.nonzero(self.labels == classLabel)[0]   # take out the index having label == classLabel
            for dim in range(dimensions):
                extractedData = self.train_data[idx, dim]
                L = np.max(extractedData)
                S = np.min(extractedData)
                G = (L - S)/(no_of_bins - 3)
                if G < 0.0001:
                    G = 0.0001
                # G will be the width of all bins, except for bin 0 and bin N-1, whose width is infinite.
                range_min = - math.inf
                range_max = S - G/2
                for bin_num in range(0, no_of_bins):
                    bin_count = 0
                    for m in range(self.train_data[idx, dim].shape[0]):
                        if(extractedData[m] >= range_min and extractedData[m] < range_max):
                            bin_count += 1
                    self.bin_min_range[classLabel-1,dim,bin_num] = range_min
                    self.bin_max_range[classLabel-1,dim,bin_num] = range_max
                    range_min = range_max
                    if bin_num == no_of_bins - 2:   # if bin == N-1, width is infinite
                        range_max =  math.inf
                    else:                           # else increase bin width by G 
                        range_max += G                        
                    # calculate bin probability
                    self.bin_probability[classLabel-1,dim,bin_num] = bin_count/(self.train_data[idx,dim].shape[0]*G)
                    # training phase
                    print("Class {:d}, attribute {:d}, bin {:d}, P(bin | class) = {:.2f}".format(classLabel,
                                                                                                   dim,
                                                                                                   bin_num,
                                                                                                   self.bin_probability[classLabel-1,dim,bin_num]))
                    
        print("\n============================================ Classification =======================================================\n")
       
        
    def test(self, no_of_bins):
        no_of_classes = len(self.classes)
        dimensions = self.dimensions-1
        # no of rows from test data and train data
        testDataSize = self.test_data.shape[0]
        trainDataSize = self.train_data.shape[0]
        test_probability = np.zeros((testDataSize,no_of_classes))
        # calculate probability of each label/class
        class_probability = np.divide(np.histogram(self.labels, range=(1,10))[0], trainDataSize)
        # to store and print accuracy
        result = np.empty((testDataSize,5))

        for i in range(testDataSize):   # iterating rows of test data
            for j in range(no_of_classes):
                probability = 1
                for dim in range(dimensions):
                    for bin_num in range(no_of_bins): 
                        if(self.test_data[i,dim] >= self.bin_min_range[j,dim,bin_num] and self.test_data[i,dim] < self.bin_max_range[j,dim,bin_num]):
                            probability = probability * self.bin_probability[j,dim,bin_num]
                test_probability[i,j] = probability * class_probability[j]
            
            test_probab_sum = np.sum(test_probability[i,:])
            for j in range(no_of_classes):
                if test_probab_sum == 0:
                    test_probability[i,j] = 1/no_of_classes
                else:
                    test_probability[i,j] = np.divide(test_probability[i,j],test_probab_sum)
            
            # calculating accuracy 
            # result[0:id, 1:predicted_class, 2:probability, 3:true_class, 4:accuracy]
            result[i,0] = i
            result[i,3] = self.test_data[i,-1]
            result[i,1], result[i,2], result[i,4] = Naive_Bayes.calc_accuracy(labels=self.classes,
                                                                              groundTruth=self.test_data[i,-1],
                                                                              probabs=test_probability[i,:]) 
        # printing classification
        Naive_Bayes.print_result(result)

class Gaussians(Naive_Bayes):
    def __init__(self, training_file, test_file):
        Naive_Bayes.__init__(self, training_file, test_file)
        
    def normpdf(self, x, mu, sigma):
        u = (x-mu)/abs(sigma)
        y = (1/(np.sqrt(2*math.pi)*abs(sigma)))*math.exp(-u*u/2)
        return y
        
    def train(self):
        no_of_classes = len(self.classes)
        dimensions = self.dimensions-1
        self.gauss_dist = np.empty((no_of_classes, dimensions, 2)) # 3D array to store mean and std of respective class 
        print("\n============================================ Training Phase =====================================================\n")
        
        for classLabel in self.classes:
            classLabel = int(classLabel)   # float to int
            idx = np.nonzero(self.labels == classLabel)[0]   # take out the index having label == classLabel
            for dim in range(dimensions):
                mean = np.mean(self.train_data[idx, dim])
                stdev = np.std(self.train_data[idx, dim])
                if stdev < 0.01:
                    stdev = 0.01
                print("Class {:d}, attribute {:d}, mean = {:.2f}, std = {:.2f}".format(classLabel, dim, mean, stdev))
                self.gauss_dist[classLabel-1, dim, 0] = mean
                self.gauss_dist[classLabel-1, dim, 1] = stdev
        print("\n============================================ Classification =======================================================\n")
        
    def test(self):
        # no of rows from test data and train data
        testDataSize = self.test_data.shape[0]
        trainDataSize = self.train_data.shape[0]
        no_of_classes = len(self.classes)
        dimensions = self.dimensions-1
        test_probability = np.zeros((testDataSize,no_of_classes))
        class_probability = np.divide(np.histogram(self.labels, range=(1,10))[0], trainDataSize)
        # to store and print accuracy
        result = np.empty((testDataSize,5))
        
        for i in range(testDataSize):
            for j in range(no_of_classes):
                probability = 1
                for dim in range(dimensions):
                    probability = probability * self.normpdf(self.test_data[i,dim],self.gauss_dist[j,dim,0],self.gauss_dist[j,dim,1])
                test_probability[i,j] = probability * class_probability[j]
            test_probab_sum = np.sum(test_probability[i,:])
            for j in range(no_of_classes):
                if test_probab_sum == 0:
                    test_probability[i,j] = 1/no_of_classes
                else:
                    test_probability[i,j] = np.divide(test_probability[i,j],test_probab_sum)
            
            # calculating accuracy 
            # result[0:id, 1:predicted_class, 2:probability, 3:true_class, 4:accuracy]
            result[i,0] = i
            result[i,3] = self.test_data[i,-1]
            result[i,1], result[i,2], result[i,4] = Naive_Bayes.calc_accuracy(labels=self.classes,
                                                                              groundTruth=self.test_data[i,-1],
                                                                              probabs=test_probability[i,:]) 
        # printing classification
        Naive_Bayes.print_result(result)

class Mixtures(Naive_Bayes):
    def __init__(self, training_file, test_file):
        Naive_Bayes.__init__(self, training_file, test_file)
        
    def normpdf(self, x, mu, sigma):
        u = (x-mu)/abs(sigma)
        y = (1/(np.sqrt(2*math.pi)*abs(sigma)))*np.exp(-u*u/2)
        return y

    def train(self, no_of_gaussians):
        iterations = 50  # stopping criterion for the EM is simply that the loop has been executed 50 times.
        no_of_classes = len(self.classes)
        dimensions = self.dimensions-1
        self.gauss_dist = np.empty((no_of_classes, dimensions, no_of_gaussians, 3)) # 4D array to store mean and std of respective class 
        print("\n============================================ Training Phase =====================================================\n")
        
        for classLabel in self.classes:
            classLabel = int(classLabel)   # float to int
            idx = np.nonzero(self.labels == classLabel)[0]   # take out the index having label == classLabel
            for dim in range(dimensions):
                extractedData = self.train_data[idx,dim]
                L = np.max(extractedData)
                S = np.min(extractedData)
                G = (L - S)/no_of_gaussians
                em_ip = np.zeros((no_of_gaussians, 3))
                for gauss in range(no_of_gaussians):
                    em_ip[gauss, 0] = S + G*gauss + G/2  # mean
                    em_ip[gauss, 1] = 1  # sigma
                    em_ip[gauss, 2] = 1/no_of_gaussians  # weight
                pdf = np.zeros((no_of_gaussians,extractedData.shape[0]))
                for iterr in range(iterations):
                    # E Steps / Estimation or expectation steps
                    for i in range(extractedData.shape[0]):
                        summation = 0
                        for j in range(no_of_gaussians):
                            pdf[j,i] = self.normpdf(extractedData[i], em_ip[j,0], em_ip[j,1])*em_ip[j,2]
                            #pdf[j,i] = mlab.normpdf(extractedData[i], em_ip[j,0], em_ip[j,1])*em_ip[j,2]
                            summation = summation + pdf[j,i]
                        if(summation == 0):
                            summation = 0.0001
                        for j in range(no_of_gaussians):
                            pdf[j,i] = pdf[j,i]/summation
                    
                    # M Steps / Maximization steps
                    denominator = np.zeros((no_of_gaussians,1))
                    for j in range(no_of_gaussians):
                        numerator = 0
                        sgma = 0
                        for i in range(extractedData.shape[0]):
                            numerator = numerator + pdf[j,i] * extractedData[i]
                            denominator[j] = denominator[j] + pdf[j,i]
                            sgma = sgma + pdf[j,i] * ((extractedData[i] - em_ip[j,0])**2)
                        em_ip[j,0] = numerator/denominator[j]   # calculating mean
                        em_ip[j,1] = np.sqrt(sgma/denominator[j])   # calculating std deviation
                        if(em_ip[j,1] < 0.01):
                            em_ip[j,1] = 0.01
                    sum_denominator = np.sum(denominator[:])
                    for j in range(no_of_gaussians):
                        em_ip[j,2] = denominator[j]/sum_denominator   # calculating weight
                for gauss in range(no_of_gaussians):
                    self.gauss_dist[classLabel-1, dim, gauss, 0] = em_ip[gauss, 0]
                    self.gauss_dist[classLabel-1, dim, gauss, 1] = em_ip[gauss, 1]
                    self.gauss_dist[classLabel-1, dim, gauss, 2] = em_ip[gauss, 2]
                    print("Class {:d}, attribute {:d}, Gaussian {:d}, mean = {:.2f}, std = {:.2f}".format(classLabel,dim,gauss,
                                                                                                          em_ip[gauss, 0],
                                                                                                          em_ip[gauss, 1]))
        
        print("\n============================================ Classification =======================================================\n")
        
    def test(self, no_of_gaussians):
        # no of rows from test data and train data
        testDataSize = self.test_data.shape[0]
        trainDataSize = self.train_data.shape[0]
        no_of_classes = len(self.classes)
        dimensions = self.dimensions-1
        test_probability = np.zeros((testDataSize,no_of_classes))
        class_probability = np.divide(np.histogram(self.labels, range=(1,10))[0], trainDataSize)
        # to store and print accuracy
        result = np.empty((testDataSize,5))
        
        for i in range(testDataSize):
            for j in range(no_of_classes):
                probability = 1
                for dim in range(dimensions):
                    avg_probability = 0
                    for gauss in range(no_of_gaussians):
                        mean = self.gauss_dist[j,dim,gauss,0]
                        stdev = self.gauss_dist[j,dim,gauss,1]
                        weight = self.gauss_dist[j,dim,gauss,2]
                        avg_probability += self.normpdf(self.test_data[i,dim],mean,stdev)*weight
                    probability = probability * avg_probability
                test_probability[i,j] = probability * class_probability[j]
            test_probab_sum = np.sum(test_probability[i,:])
            for j in range(no_of_classes):
                if test_probab_sum == 0:
                    test_probability[i,j] = 1/no_of_classes
                else:
                    test_probability[i,j] = np.divide(test_probability[i,j],test_probab_sum)
            
            # calculating accuracy 
            # result[0:id, 1:predicted_class, 2:probability, 3:true_class, 4:accuracy]
            result[i,0] = i
            result[i,3] = self.test_data[i,-1]
            result[i,1], result[i,2], result[i,4] = Naive_Bayes.calc_accuracy(labels=self.classes,
                                                                              groundTruth=self.test_data[i,-1],
                                                                              probabs=test_probability[i,:]) 
        # printing classification
        Naive_Bayes.print_result(result)

def main():
    
    ''' User input should follow the given format:
        -----------------------------------------------------------------------------------------
        naive_bayes <training_file> <test_file> histograms <number>
        naive_bayes <training_file> <test_file> gaussians
        naive_bayes <training_file> <test_file> mixtures <number>
        -----------------------------------------------------------------------------------------
        The arguments provide to the program the following information:
        > The first argument is the path name of the training file
        > The second argument is the path name of the test file
        > The third argument can have three possible values: histograms, gaussians, or mixtures.
            > If the third argument is histograms: 
                > The fourth argument specifies how many bins to use for each histogram. 
            > If the third argument is mixtures: 
                > The fourth argument specifies how many Gaussians to use for each mixture.                

        # Sample input -->

        1) Histograms:
        naive_bayes yeast_training.txt yeast_test.txt histograms 7

        2) Gaussians:
        naive_bayes yeast_training.txt yeast_test.txt gaussians

        3) Mixtures:
        naive_bayes yeast_training.txt yeast_test.txt mixtures 3
        
    '''
    print("\nPlease provide the training & test data's file path, classification method using the following format:\n",
	"naive_bayes <training_file> <test_file> histograms <number>\n",
	"naive_bayes <training_file> <test_file> gaussians\n",
	"naive_bayes <training_file> <test_file> mixtures <number>\n\n",
	"Please enter here:")
    user_input = input()   # user input in the aforementioned fashion
    input_list = user_input.split()   # converting the input into a list
    if(input_list[3] == 'histograms'):
        histogram = Histograms(training_file=input_list[1], test_file=input_list[2])
        histogram.load_data()
        histogram.train(no_of_bins=int(input_list[4]))
        histogram.test(no_of_bins=int(input_list[4]))
    elif(input_list[3] == 'gaussians'):
        gaussian = Gaussians(training_file=input_list[1], test_file=input_list[2])
        gaussian.load_data()
        gaussian.train()
        gaussian.test()
    elif(input_list[3] == 'mixtures'):
        gaussian_mixture = Mixtures(training_file=input_list[1], test_file=input_list[2])
        gaussian_mixture.load_data()
        gaussian_mixture.train(no_of_gaussians=int(input_list[4]))
        gaussian_mixture.test(no_of_gaussians=int(input_list[4]))
    else:
        print("User input Error!\nPlease check again, or kindly refer the README file.")
    
main()