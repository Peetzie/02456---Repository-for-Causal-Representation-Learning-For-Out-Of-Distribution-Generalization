#iVAE class
import torch
from torch import nn, Tensor
from torch.distributions import Distribution
from typing import *

from ReparameterizedDiagonalGaussian import ReparameterizedDiagonalGaussian

def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)

class iVAE(nn.Module):
    # An identifiable Variational Autoencoder
    #The prior is conditioned on the environment, E, and target label Y through a Neural network.
    
    def __init__(self, input_shape:torch.Size, latent_features:int) -> None:
        super(iVAE, self).__init__()
        
        self.input_shape = input_shape
        self.latent_features = latent_features
        
        '''
        The iVAE consist of 5 NNs which are intended to follow the general structure presented by Lu et al 2022. Where we've simply used a network for training
        only the natural parameters of the exponential family prior, rather than training both natural parameters and sufficient statistics as was done for the NF-iVAE by Lu et al 2022:
        
        1. lambdaf prior
        
        2. X-encoder (Classic image CNN)
        3. (Y, E)-encoder
        4. (X, Y, E)-merger/encoder

        5. Decoder

        1: Learn prior based on the label distribution for the given environment
        2-4: Encoding X, encoding Y and E and merging these two encoders, to generate a 
             qz which is conditional on the environment.
        5: Decodes the latent space through pz. Since the latent space now contain some measure
           of environment, then this distribution pz is consequentially conditioned on the environment
        '''
        #### PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS ####
        #NN 1/5
        self.Lambdaf_prior = nn.Sequential(
            nn.Linear(in_features = 2, out_features=50), #Input
            nn.Linear(in_features = 50, out_features=50), #Fully connected
            nn.ReLU(),
            nn.Linear(in_features = 50, out_features = 20))  #Output
        #### PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS ####

         #For NN 2/5 X-Encoder
        self.encoderCNN1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        self.encoderCNN2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        self.encoderCNN3 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        ##NN 3/5 (Y, E)-Encoder
        self.YEencoder = nn.Sequential(
            nn.Linear(in_features = 2, out_features=100),
            nn.Linear(in_features = 100, out_features = 100),
            nn.ReLU())

        ##NN 4/5 (X, Y, E)-merger/encoder
        self.XYEmerger = nn.Sequential(
            nn.Linear(in_features = 32 * 4 * 4 + 100, out_features=100),
            nn.Linear(in_features = 100, out_features = 100),
            nn.ReLU(),
            nn.Linear(in_features = 100, out_features = 2*latent_features))


        #For NN 5/5 (Decoder)
        self.decoderFFN = nn.Linear(in_features=latent_features, out_features = 32 * 4 * 4)
        self.decoderFFN2 = nn.Linear(in_features = 32 * 4 * 4, out_features = 32 * 4 * 4)
        
        self.decoderCNN1 = nn.ConvTranspose2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1
                              ,output_padding = 0)
        self.decoderCNN2 = nn.ConvTranspose2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1
                              ,output_padding = 1)
        self.decoderCNN3 = nn.ConvTranspose2d(in_channels = 32, out_channels = 3, kernel_size = 3, stride = 2, padding = 1
                              ,output_padding = 1)

    #The prior parameters are now estimated through a neural network instead of just assuming a standard gaussian distribution (seen in the VAE.py script)
    def prior_params(self, y, e):
        ye = torch.cat((y, e), dim = 1)
        ye = ye.to(torch.float32)
        lambdaf_parameters = self.Lambdaf_prior(ye) #NN 1/5

        return lambdaf_parameters

    #Encoder and decoder functions which can be modified to include both a CNN and an ordinary FFN  
    def encoder(self, x, y, e):
        #CNN (X-encoder) NN 2/5
        x = self.relu(self.encoderCNN1(x))
        x = self.relu(self.encoderCNN2(x))
        x = self.relu(self.encoderCNN3(x))

        #Flatten so it can be used as input in the merged FFN
        x = x.view(x.size(0), -1)

        #FFN (Y,E)-encoder, NN 3/5
        ye = torch.cat((y, e), dim = 1)
        ye = ye.to(torch.float32)
        ye = self.YEencoder(ye)

        #Merged encoder
        xye = torch.cat((x,ye), dim = 1)
        xye = self.XYEmerger(xye) #NN 4/5
    
        return xye
   
    #NN 5/5
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
        
    def posterior(self, x:Tensor, y:Tensor, z:Tensor) -> Distribution:
        #Calculate the parameters in the distribution: q(z|x) ~ N()
        h_x = self.encoder(x, y, z)
        mu, log_sigma =  h_x.chunk(2, dim=-1)
        
        #Return a latent space probability distribution given the input image x.
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def prior(self, y, e)-> Distribution:
        #The prior distribution, which in case of the iVAE is a normal distribution with conditioning of 
        # the mean and variance on Y and E through a neural network with Y and E as input.

        prior_params = self.prior_params(y, e)
        #chunks the prior params into two vectors for mu and log sigma respectively.
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def observation_model(self, z:Tensor) -> Distribution:
        #Decoder. 
        #Return the probability of observing our input X, given our constructed latent space p(X|Z)
        px_means = self.decoder(z)
        px_means = px_means.view(-1, *self.input_shape)
        log_var = 0.01 * torch.ones(px_means.shape)
        log_sigma = torch.log(torch.sqrt(log_var))
        return ReparameterizedDiagonalGaussian(mu = px_means, log_sigma = log_sigma)
        

    def forward(self, x, y, e) -> Dict[str, Any]:
        #Forward pass through all steps of the VAE model.
        #### Run through ENCODER and calculate mu and sigma for latent space sampling
        # define the posterior q(z|x)
        qz = self.posterior(x, y, e)
        
        #### LATENT SPACE sampling using the reparameterization trick
        z = qz.rsample()
        
        #### DECODE
        px = self.observation_model(z)
        
        # define the prior p(z)
        #(Used for kullback-leibler divergence calculation in the ELBO loss)
        pz = self.prior(y,e)
        
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}
    
    def reduce(self, x:Tensor) -> Tensor:
        """for each datapoint: sum over all dimensions"""
        return x.view(x.size(0), -1).sum(dim=1)

    def VariationalInference(self, x, y, e, beta):
        self.beta = beta
        
        # Get the probability distributions px, pz, qz and the used latent space sample z.
        outputs = self.forward(x, y, e)
        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]
        
        # evaluate log probabilities
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