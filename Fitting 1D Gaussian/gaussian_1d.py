import numpy as np

def load_data(fname):
#    fname = "yeast_training.txt"
    input_file = open(fname, "r")
    lines = input_file.readlines()   # reading data file
    dimensions = len(list(filter(None, lines[1].split(" ")))) - 1  # geting attribute length
    train_data = np.zeros((1,dimensions))   # declaring empty array to stack the data
    labels = np.zeros((1,1))   # same for the labels
    for line in lines:
        x = line.split(" ")
        x = list(filter(None, x))   # getting rid of empty values
        x = list(map(float, x))    # keeping the relative order between the data values
        train_data = np.vstack((train_data, x[:-1]))
        labels = np.vstack((labels, x[-1]))
    return train_data[1:], labels[1:], dimensions

def calculate_mean(data, dim):
    if data.shape[0] == 0:
        return 0
    else:
        return np.sum(data[:, dim])/data.shape[0]

def calculate_variance(data, mean, dim):
    if data.shape[0] == 0:
        return 0
    else:
        return np.sum(np.square(data[:, dim] - mean))/(data.shape[0] - 1)

	
	
print("Please enter the path of the dataset file: ")
file_name = input()
data, labels, dimensions = load_data(file_name)

ind = np.unravel_index(np.argsort(labels, axis=None), labels.shape)   # indices of the sorted labels 

n_lbls = np.unique(labels)
for lbl in n_lbls:
    data_chunk = data[np.nonzero(labels == int(lbl))[0],:]   # take out the data having same labels
    for dim in range(dimensions):
        mean = calculate_mean(data_chunk, dim)
        var = calculate_variance(data_chunk, mean, dim)
        print("Class {:d},".format(int(lbl)), " dimension {:d}".format(dim+1), " mean = {:.2f}".format(mean), " variance = {:.2f}".format(var))

