# Testing Calibration in Nearly-Linear Time

The most important methods can be found in min_flow.ipynb. In min_flow.ipynb, we have methods using min-cost flow solver from Gurobi and our own Dynamic Programming method. These methods are used throughout our experiments.

Specifically, we use smCE.py or smCE.ipynb for section 6.1 in our paper, which compares smCE, LDTC, and convolved ECE(smECE).

Section 6.2 experiments can be found in the temperature folder. This folder contains train.py, which trains DenseNet-40 on CIFAR-100, and test.py, which compares smCE errors between the original model, temperature scaling, and isotonic regression.

Section 6.3 experiments can be found in files min_flow.ipynb and min_flow_no_np.py. The first file includes calculating smooth calibration error using CVXPY, the min-cost flow solver from Gurobi, and our own Dynamic Programming method. The second file contains our own Dynamic Programming method without using numpy, which means we can run it with pypy and improve it to be the fastest way to calculate smooth calibration error.
