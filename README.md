## Implementation of Moving Semantic Transfer Network
[explanation](https://sally20921.github.io/doc/DA/MSTN.pdf)

### Ultimate goal is to  develop a deep NN  that is able to predict labels for the  samples from taraget domain 

### Besides the 1) standard source classification loss, we also employ 2)domain adversarial loss(domain confusion loss)  3)semantic loss (pseudo-labeled semantic loss)
###
- G :  Feature Extractor
- D : Domain Discriminator (whether features from G arise from source  or  target domain) 
- F : Classifier  
### Implementation Detail
- CNN architecture :  AlexNet Architecture, a bottleneck layer fcb with  256 units is added after fc7 layer  for  safer  transfer representation  learning.
- fcb as inputs to the  discriminator  as well as centroid  computation
- discriminator: x-> 1024 -> 1024 -> 1, dropout is used 
- Hyper-parameters tuning : weight balance parameter lamda, and moving average coefficient theta. (reverse  validation) 
- Stochastic Gradient Descent  with 0.9 momentum is used. 

### MSTN pseudo-code 
- feature extraction
- align distributions (moving  average) 
- centroid alignment  
