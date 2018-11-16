# Cifar10-Classification
Requirements- \
python 3.6.6 \
Scikit-learn 0.20.0 \
keras 2.2.2 \
tensorflow 1.9.0

The dataset files should be downloaded from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz. \
Extract and rename the folder to cifar10. 

To run a classifier -

python main.py --classifier DT --depth 3 --representation MLP_Embedding \
python main.py --clasifier SVM --C 200.0 --representation LDA \
python main.py --classifier LR --eta 0.001 --epochs 100 --batch_size 1000 \
python main.py --classifier MLP --eta 0.0001 --epochs 10000 --batch_size 100 --dropout 0.2

All the above mentioned parameters have some default value.

The PCA would by default use 350 number of components. To use some other number, use the --ncomponents option along with --representation PCA \
python main.py --classifier DT --representation PCA --ncomponents 100 

At present, I have pre calculated LDA and PCA on the training data and saved that in the data folder. 
To recalculate it while running a classifier, 

python main.py --create_dim_red PCA\
python main.py --create_dim_red LDA
