## implementation of Moving Semantic Transfer Network

### Besides the 1) standard source classification loss, we also employ 2)domain adversarial loss


### Implementation Detail
- CNN architecture :  AlexNet Architecture, a bottleneck layer fcb with  256 units is added after fc7 layer  for  safer  transfer representation  learning.
- fcb as inputs to the  discriminator  as well as centroid  computation
- discriminator: x-> 1024 -> 1024 -> 1, dropout is used 
- Hyper-parameters tuning : 
