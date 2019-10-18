# Uncertainty-Estimation-BNN
## The Data
[California Housing Prices](https://www.kaggle.com/camnugent/california-housing-prices)  
## Types of Uncertainty
`Epistemic Uncertainty`: accounts for uncertainty in the model -- uncertainty which can be explained away given enough data （e.g., in the problem of dangerous creatures predicton, if only given the picture of lion and giraffe in the training phase, the model will have high uncertainty of classify zombie included test dataset.  
![e1](https://github.com/yizhanyang/Uncertainty-Estimation-BNN/blob/master/e1.jpg)  
  
`Aleatoric uncertainty`: captures noise inherent in the observations. (e.g, in the example above, probably not dangerous if they are not hungry).  
## Method
`MC Dropout`: Don’t change structure of NN,   
Dropout: weight = 0,   
Probability = drop w/ total w,  
Open drop out during prediction,  
Deviation between confidence and hit probabilty = 0.1656。  
[Original Paper: Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142) 
