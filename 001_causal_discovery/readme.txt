+++++environment configuration++++++


The code is only tested on Linux based System (Ubuntu 20.04). 
The python version is 3.6.9. The pytorh version is 1.2.0 with GPU acceleration. 

It is unknown that if the code is compatible on windows or different versions of pytorh and python. 
We have not tested to run our code in a CPU environment. 
To avoid errors caused by inconsistent environment, you are encouraged to run our code under a same environment.

For cifar10, the clustering method used is SPICE. We use the pretrained model SPICE-Self*. The model and source code are downloaded via (https://github.com/niuchuangnn/SPICE)

+++++quick start++++++

sudo chmod 755 *.sh

#########run xyguassian dataset################
./run_yxguassain.sh

#########run yxguassian dataset################
./run_yxguassain.sh

#########run other uci datasets################
./run_uci.sh

#########run mnist################
./run_mnist.sh


#########run cifar10################
./run_cifar10.sh







