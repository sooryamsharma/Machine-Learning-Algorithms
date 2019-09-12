import numpy as np

class DataHandler:
    def __init__(self, training_file, test_file, option, pruning_thr):
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

class node():
    """Class for basic node functionality"""
    def __init__(self):
        self.tree_id = 1
        self.node_id = 0
        self.attr = -1
        self.thr = -1
        self.gain = 0
        self.value = None
        self.left_child = None
        self.right_child = None
        
    def print_node(self):
        print("tree={:2d}, node={:3d}, feature={:2d}, thr={:6.2f}, gain={:f}".format(tree_id, self.node_id, 
                                                                                       self.attr, self.thr, self.gain))
    
    def classify(self, row):
        if(self.value != None):
            return self.value
        else:
            if (row[self.attr] < self.thr):
                return self.left_child.classify(row)
            else:
                return self.right_child.classify(row)
        
class DecisionTrees(DataHandler):
    def __init__(self, training_file, test_file, option, pruning_thr):
        DataHandler.__init__(self, training_file, test_file, option, pruning_thr)
        self.pruning_thr = pruning_thr
        self.option = option
        
            
    def print_tree_bfs(self, tree):
        queue = []
        queue.append(tree)
        while len(queue) != 0:
            root = queue.pop(0)
            root.print_node()
            
            if (root.left_child and root.right_child != None):
                queue.append(root.left_child)
                queue.append(root.right_child)
            elif (root.left_child != None and root.right_child == None):
                queue.append(root.left_child)
            elif (root.left_child == None and root.right_child != None):
                queue.append(root.right_child)
        
        for node in queue:
            node.print_node()     
        
    def dtree(self):
        train_data = self.train_data
        pruning_thr = self.pruning_thr
        test_data = self.test_data
        ground_truth = test_data[:,-1]
        
        classes = np.unique(train_data[:,-1])
        attributes = [dim for dim in range(train_data.shape[1]-1)]
        counts = self.class_count(train_data, classes)
        
        print("===========================Training Phase==================================")
        
        global tree_id, node_id
        node_id = 1

        if (self.option == "optimized"):
            tree_id = 0
            tree = self.getDT(train_data, attributes, counts, pruning_thr, classes, randomize=False)
            self.print_tree_bfs(tree)
            print("===========================Classification Phase==================================")
            count = 0
            for i in range(test_data.shape[0]):
                res = tree.classify(test_data[i])
                if isinstance(res, dict):
                    res = max(res.keys(), key=(lambda k: res[k]))
                acc = 0
                if res == ground_truth[i]:
                    count += 1
                    acc = 1
                print("ID={:5d}, predicted={:3d}, true={:3d}, accuracy={:4.2f}".format(i+1, int(res), int(ground_truth[i]), acc))
            print("classification accuracy={:6.4f}".format(count / len(ground_truth)))
            
        elif (self.option == "randomized"):
            tree_id = 0
            tree = self.getDT(train_data, attributes, counts, pruning_thr, classes, randomize=True)
            self.print_tree_bfs(tree)
            print("===========================Classification Phase==================================")
            count = 0
            for i in range(test_data.shape[0]):
                res = tree.classify(test_data[i])
                if isinstance(res, dict):
                    res = max(res.keys(), key=(lambda k: res[k]))
                acc = 0
                if res == ground_truth[i]:
                    count += 1
                    acc = 1
                print("ID={:5d}, predicted={:3d}, true={:3d}, accuracy={:4.2f}".format(i+1, int(res), int(ground_truth[i]), acc))
            print("classification accuracy={:6.4f}".format(count / len(ground_truth)))
            
        elif (self.option.find("forest") != -1):
            no_of_trees = int(self.option[len("forest"):])
            forest = []
            for tree_no in range(no_of_trees):
                tree_id = tree_no
                node_id = 1
                tree = self.getDT(train_data, attributes, counts, pruning_thr, classes, randomize=True)
                forest.append(tree)
                self.print_tree_bfs(tree)
            print("===========================Classification Phase==================================")
            count = 0
            for i in range(test_data.shape[0]):
                forest_res = [tree.classify(test_data[i]) for tree in forest] 
                best_res = []
                for res in forest_res:
                    if isinstance(res, dict):
                        res = max(res.keys(), key=(lambda k: res[k]))
                    best_res.append(res)
                res = max(set(best_res), key=best_res.count)
                acc = 0
                if res == ground_truth[i]:
                    count += 1
                    acc = 1
                print("ID={:5d}, predicted={:3d}, true={:3d}, accuracy={:4.2f}".format(i+1, int(res), int(ground_truth[i]), acc))
            print("classification accuracy={:6.4f}".format(count / len(ground_truth)))
            
    def class_count(self, data, classes):            
        """Counts the number of each type of example in a dataset."""
        count={}
        for lbl in range(len(classes)):
            count[lbl] = data[data[:,-1] == lbl].shape[0]/data.shape[0]
        return count

    def getDT(self, data, attributes, counts, pruning_thr, classes, randomize):
        global tree_id, node_id
        if (data.shape[0] < pruning_thr):
            tree = node()
            tree.value = counts
            tree.tree_id = tree_id
            tree.node_id = node_id
            node_id += 1
            return tree
    
        elif (np.unique(data[:,-1]).shape[0] == 1):
            tree = node()
            tree.value = data[0][-1]
            tree.tree_id = tree_id
            tree.node_id = node_id
            node_id += 1
            return tree
    
        else:
            tree = node()
            best_attr, best_thr, best_gain = self.find_best_split(data, attributes, classes, randomize)
            tree.attr = best_attr
            tree.thr = best_thr
            tree.gain = best_gain
            tree.tree_id = tree_id
            tree.node_id = node_id
            node_id += 1
            left_tree = data[data[:, best_attr] < best_thr, :]
            right_tree = data[data[:, best_attr] >= best_thr, :]
            tree.left_child = self.getDT(left_tree, attributes, self.class_count(data, classes), pruning_thr, classes, randomize)
            tree.right_child = self.getDT(right_tree, attributes, self.class_count(data, classes), pruning_thr, classes, randomize)
            return tree
        
    def find_best_split(self, data, attributes, classes, randomize):
        """Find the best attribute to split by iterating over every feature / value
           and calculating the information gain."""
        best_gain = -1  # keep track of the best information gain
        best_attr = -1  # keep train of the feature / value that produced it
        best_thr = -1
        n_features = len(attributes)
        
        if (randomize == False):
            for col in range(n_features):  # for each feature
                # unique values in the column
                values = data[:,col]   
                min_value = np.min(values)
                max_value = np.max(values)
                for iterr in range(self.pruning_thr):
                    thr = min_value + iterr * (max_value - min_value)/(self.pruning_thr+1)
                    gain = self.info_gain(data, col, thr, classes) # Calculate the information gain from this split
                    if gain > best_gain:
                        best_gain, best_attr, best_thr = gain, col, thr
    
        elif (randomize == True):
            # unique values in the column
            rndm_col = np.random.choice(np.array(attributes), replace=False) 
            values = data[:,rndm_col]
            min_value = np.min(values)
            max_value = np.max(values)
            for iterr in range(self.pruning_thr):
                thr = min_value + iterr * (max_value - min_value)/(self.pruning_thr+1)
                gain = self.info_gain(data, rndm_col, thr, classes) # Calculate the information gain from this split
                if gain > best_gain:
                    best_gain, best_attr, best_thr = gain, rndm_col, thr
        return best_attr, best_thr, best_gain
    
    def info_gain(self, data_sample, attribute, thresh, classs):
        """Information Gain. 
        The uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        """
        left = data_sample[data_sample[:,attribute] < thresh, :]
        right = data_sample[data_sample[:,attribute] >= thresh, :]
        no_of_sampls = len(data_sample)
        entropy, entropy_left, entropy_right = 0, 0, 0
        _e_ = np.exp(-10)
        for i in range(len(classs)):
            p = len(data_sample[data_sample[:,-1] == classs[i]])/(data_sample.shape[0]+_e_)
            if p > 0:
                entropy += -p*np.log2(p) 
            p_left = len(left[left[:,-1] == classs[i]])/(left.shape[0]+_e_)
            if p_left > 0:
                entropy_left += -p_left*np.log2(p_left) 
            p_right = len(right[right[:,-1] == classs[i]])/(right.shape[0]+_e_)
            if p_right > 0:
                entropy_right += -p_right*np.log2(p_right)
        gain = entropy - (len(left)/no_of_sampls)*entropy_left - (len(right)/no_of_sampls)*entropy_right
        return gain

def main():
    
    ''' User input should follow the given format:
        ---------------------------------------------------------------------------------------------
        dtree <training_file> <test_file> <option> <pruning_thr>
        ---------------------------------------------------------------------------------------------
        The arguments provide to the program the following information:
        > The first argument, <training_file>, is the path name of the training file, where the training data is stored. 
          The path name can specify any file stored on the local computer.
        > The second argument, <test_file>, is the path name of the test file, where the test data is stored. 
          The path name can specify any file stored on the local computer.
        > The third argument, <option>, can have four possible values: optimized, randomized, forest3, or forest15. 
          It specifies how to train (learn) the decision tree, and will be discussed later.
        > The fourth argument, <pruning_thr>, is a number greater than or equal to 0, that specifies the pruning threshold.

        # Sample inputs -->
        dtree pendigits_training.txt pendigits_test.txt optimized 50
        dtree pendigits_training.txt pendigits_test.txt randomized 50
        dtree pendigits_training.txt pendigits_test.txt forest3 50
    '''

    print("\nPlease provide the following: training data path, test data path, training option, pruning threshold:\n",
    "dtree <training_file> <test_file> <option> <pruning_thr>\n",
    "Please enter here:")
    user_input = input()   # user input in the aforementioned fashion
    input_list = user_input.split()   # converting the input into a list
    # creating object
    dtree = DecisionTrees(training_file=input_list[1], test_file=input_list[2], option=input_list[3], pruning_thr=int(input_list[4]))
    dtree.load_data()
    dtree.dtree()

main()
