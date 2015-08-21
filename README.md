# Maximum_Entroy_Model

This is a simple implementation of Maximum Entropy Model. I used the same training data and project structure as Naive Bayes Model. The training data are blogs. This model aims to predicting the genders of the authors of blog posts.

This is my second python projects. There are still many interesting problems here. If I implement a MaxEnt model now, I will do it using the following steps. 

1. Transform each blog to a binary vector. Each value of the vector represents the occurence of words in bag-of-words model. The size of each vector is 1*N, which N is the number of words in bag-of-words model. The vector would be sparse.

2. Initiate lambda(feature weight). The size of lambda is 1*2N (f(M, word), f(W, word)).

3. Implement functions for calculating log-likelihood and gradients by using matrix calculation.

4. Pass log-likelihood function, gradients function and initial lambda into 'scipy.optimize.fmin_l_bfgs_b' function, which is a black box function for getting 'minimum'. Now we get the optimum value for lambda.

5. In test part, calculate likelihood of M and W. Then get the argmax_y(P(y|X)).
