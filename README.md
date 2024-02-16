Most important methods can be found in demo.ipynb. In demo, We have methods for S_0 and S_{base} and our box simplex implementation. These methods are used through out our experiments.

Specificly, we use smCE.py or smCE.ipynb for section 1 in our paper which compared smCE, LDTC and convCE(smECE).

Section 2 experiments can be found in folder temperature. This folder contains train.py which train DenseNet-40, on Cifar100 and test.py which compares smCE error between original model, temperature scaling and isotonic regression.

Section 3 experiments can be found in file box.py which contrains box simplex method for both S_0 and S_{base}.
