Task 1

Requirements:
1.) numpy

2.) math

Execution:
1.) Open Terminal

2.) change directory to the current(assignment) folder.

3.) Type "python bayes_classifiers.py" and press Enter.

4.) Enter the the training & test data's file path, classification method using the following format:

1. naive_bayes <training_file> <test_file> histograms <number>

2. naive_bayes <training_file> <test_file> gaussians

3. naive_bayes <training_file> <test_file> mixtures.

## Example sample input -->

1) Histograms:
naive_bayes yeast_training.txt yeast_test.txt histograms 7

2) Gaussians:
naive_bayes yeast_training.txt yeast_test.txt gaussians

3) Mixtures:
naive_bayes yeast_training.txt yeast_test.txt mixtures 3


