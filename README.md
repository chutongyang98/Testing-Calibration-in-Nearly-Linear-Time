# Testing Calibration in Subquadratic Time

The most important methods can be found in demo.ipynb. In demo.ipynb, We have methods for S<sub>0</sub> and S<sub>base</sub> and our box simplex implementation. These methods are used throughout our experiments.

Specifically, we use smCE.py or smCE.ipynb for section 6.1 in our paper which compared smCE, LDTC, and convolved ECE(smECE).

Section 6.2 experiments can be found in folder temperature. This folder contains train.py, which trains DenseNet-40 on Cifar100, and test.py, which compares smCE errors between the original model, temperature scaling, and isotonic regression.

Section 6.3 experiments can be found in file box.py, which contains the box simplex method for both S<sub>0</sub> and S<sub>base</sub>.
