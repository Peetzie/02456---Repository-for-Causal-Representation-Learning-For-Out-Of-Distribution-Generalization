#VAE class
import torch
import numpy as np
from torch import nn, Tensor
from torch.distributions import Distribution
from torch.distributions import Normal
from typing import *

from ReparameterizedDiagonalGaussian import ReparameterizedDiagonalGaussian

def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)

class VariationalAutoencoder(nn.Module):
    #A Variational Autoencoder with
    #Based on a Gaussian unconditional prior `p(Z) = N(0, I)`
    
    def __init__(self, input_shape:torch.Size, latent_features:int) -> None:
        super(VariationalAutoencoder, self).__init__()
        
        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)
        
        # Prior parameters, chosen as p(z) = N(0, I)
        ## setting the prior to a vector consisting of zeros with dimensions (1,2*latent_features)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features]))) #From VAE class from the course.
        
        # NETWORK ARCHITECTURE BASED ON THE ARTICLE BY LU ET AL 2022.
        #### ENCODER
        self.encoderCNN1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        self.encoderCNN2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        self.encoderCNN3 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        
        self.encoderFFN2 = nn.Linear(in_features = 32 * 4 * 4, out_features=32 * 4 * 4)
        self.encoderFFN = nn.Linear(in_features = 32 * 4 * 4, out_features=2*latent_features)
        
        #Relu activation function
        self.relu = nn.ReLU()
        
        #### DECODER
        self.decoderFFN = nn.Linear(in_features=latent_features, out_features = 32 * 4 * 4)
        self.decoderFFN2 = nn.Linear(in_features = 32 * 4 * 4, out_features = 32 * 4 * 4)
        
        self.decoderCNN1 = nn.ConvTranspose2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1
                              ,output_padding = 0)
        self.decoderCNN2 = nn.ConvTranspose2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1
                              ,output_padding = 1)
        self.decoderCNN3 = nn.ConvTranspose2d(in_channels = 32, out_channels = 3, kernel_size = 3, stride = 2, padding = 1
                              ,output_padding = 1)
        
    #Encoder and decoder functions which can be modified to include both a CNN and an ordinary FFN    
    def encoder(self, x):
        #CNN
        x = self.relu(self.encoderCNN1(x))
        x = self.relu(self.encoderCNN2(x))
        x = self.relu(self.encoderCNN3(x))
        
        #Flatten so it can be used as input in the FFN
        x = x.view(x.size(0), -1)
        
        #FFN
        x = self.relu(self.encoderFFN2(x))
        x = self.encoderFFN(x)
        return x
   
    def decoder(self, z):
        #FFN decoder
        x = self.relu(self.decoderFFN(z))
        x = self.relu(self.decoderFFN2(x))
        
        # reshape x back to 'image shape' consisting of channels and pixel width/height
        x = x.view(-1, 32, 4, 4)
        
        #CNN decoder using ConvTranspose2d to inverse the CNN encoding from the encoder function
        x = self.relu(self.decoderCNN1(x))
        x = self.relu(self.decoderCNN2(x))
        x = self.decoderCNN3(x)
        return x
        
    def posterior(self, x:Tensor) -> Distribution:
        #Calculate the parameters in the distribution: q(z|x) ~ N()

        h_x = self.encoder(x)
        mu, log_sigma =  h_x.chunk(2, dim=-1)
        #Return a latent space probability distribution given the input image x.
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def prior(self, batch_size:int=1)-> Distribution:
        #The prior distribution, which in case of the VAE is simply assumed to be standard gaussian.

        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        #chunks the prior params into two vectors for mu and log sigma respectively.
        #both vectors simply contain zeros
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        #Notice that since it is log_sigma it'll become 1, when input into the reparameterized diagonal guassian function
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def observation_model(self, z:Tensor) -> Distribution:
        #Decoder. 
        #Returen the probability of observing our input X, given our constructed latent space p(X|Z)
        px_loc = self.decoder(z)
        px_loc = px_loc.view(-1, *self.input_shape)
        shape = torch.ones(px_loc.shape)
        shape = 0.1 * shape #given by the article by Lu et al 2022.
        return Normal(loc=px_loc, scale = shape)
        

    def forward(self, x) -> Dict[str, Any]:
        #Forward pass through all steps of the VAE model.
        #### Run through ENCODER and calculate mu and sigma for latent space sampling
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)
        
        #### LATENT SPACE sampling using the reparameterization trick
        z = qz.rsample()
        
        #### DECODE
        px = self.observation_model(z)
        
        # define the prior p(z)
        #(Used for kullback-leibler divergence calculation in the ELBO loss)
        pz = self.prior(batch_size=x.size(0))
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}
    
    
    def reduce(self, x:Tensor) -> Tensor:
        #Sum
        return x.view(x.size(0), -1).sum(dim=1)

    def VariationalInference(self, x, beta):
        self.beta = beta

        # Get the probabilities px, pz, qz and the used latent space sample z.
        outputs = self.forward(x)
        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]
        
        #log probabilities
        log_px = self.reduce(px.log_prob(x)) #log(p(x|z)): log probability of observing our input x, given our latent space - how good is the model recontruction?
        log_pz = self.reduce(pz.log_prob(z)) #log(p(z)): log probability of observing our latent space under the assumption that it should follow a standard gaussian distribution
        log_qz = self.reduce(qz.log_prob(z)) #log(q(z|x)): log probability of generating this latent space, given our posterior distribution
        
        # ELBO
        #########################################################################################################
        # Reconstruction loss: E_q [ log p(x|z) ]
        # Kullback-leibler divergence and beta-regularization: \beta * D_KL(q(z|x) | p(z))`
        #########################################################################################################
        kl = log_qz - log_pz
        elbo = log_px - kl
        beta_elbo = log_px - self.beta * kl
        
        # loss
        loss = -beta_elbo.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': elbo, 'log_px':log_px, 'kl': kl}
            
        return loss, diagnostics, outputs