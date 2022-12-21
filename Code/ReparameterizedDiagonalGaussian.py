#Reparameterized diagonal gaussian
import torch
import math 
from torch.distributions import Distribution
from torch import nn, Tensor


class ReparameterizedDiagonalGaussian(Distribution):
    #A normal distribution compatible with the reparameterization trick.
    
    def __init__(self, mu: Tensor, log_sigma:Tensor):
        assert mu.shape == log_sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = log_sigma.exp()
        
    def sample(self) -> Tensor:
        #sample without reparametrization trick
        return torch.normal(mean = self.mu, std = self.sigma)
        
    def rsample(self) -> Tensor:
        #sample with reparametrization trick
        eps = torch.empty_like(self.mu).normal_()
        return self.mu + self.sigma * eps
    
    def log_prob(self, z:Tensor) -> Tensor:
        #Calculate log-probability of observing z
        return - ((z - self.mu)**2)/(2*self.sigma**2) - torch.log(self.sigma) - math.log(math.sqrt(2 * math.pi)) # <- your code
    
    def mu(self):
        return(self.mu, self.sigma)