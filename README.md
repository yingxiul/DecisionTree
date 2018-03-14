This project built a Decision Tree classifier, which could solve binary classification problem. Run the code with following command:
$ python decisionTree.py [args...]

Where [args...] is a placeholder for 6 command-line arguments:
1. <train_input> : path to the training input .csv file
2. <test_input> : path to the test input .csv file
3. <max_depth> : max depth to which the tree will be built
4. <train_out> : path of output .labels file to which the predictions on the training data will be written
5. <test_out> : path of output .labels file to which the predictions on the test data will be written
6. <metrics_out> : path of output .txt file to which training error and test error will be written

For example:
$python decitionTree.py small_train.csv small_test.csv 2 train_out.labels test_out.labels metrics_out.txt
