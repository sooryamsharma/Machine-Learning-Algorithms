Task 1

Requirements:
1.) numpy

Execution:
1.) Open Terminal

2.) change directory to the current(assignment-7) folder.

3.) Type "python decision_trees.py" and press Enter.

4.) Enter the the training file's path, test file's path, option and pruning threshold using the following format:

dtree <training_file> <test_file> <option> <pruning_thr>

## Sample input -->

dtree pendigits_training.txt pendigits_test.txt optimized 50
dtree pendigits_training.txt pendigits_test.txt randomized 50
dtree pendigits_training.txt pendigits_test.txt forest3 50
dtree pendigits_training.txt pendigits_test.txt forest15 50