#iVAE class for synthetic data
import torch
from torch import nn, Tensor
from torch.distributions import Distribution
from functools import reduce
from torch.distributions import Normal
from typing import *
from ReparameterizedDiagonalGaussian import ReparameterizedDiagonalGaussian


class iVAE_synth(nn.Module):
    
    def __init__(self, input_shape:torch.Size, latent_features:int) -> None:
        super(iVAE_synth, self).__init__()
        
        self.input_shape = input_shape
        self.latent_features = latent_features
        
        '''
        According to page 31-32 the iVAE consist of 3 NNs:
        1. lambdaf prior

        2. (X, Y, E)-merger/encoder

        3. Decoder

        1: Learn priors based on the label distribution for the given environment
        2: Encoding X, Y and E merged together, to generate a 
             qz which is conditional on the environment.
        3: Decodes the latent space through pz. Since the latent space now contain some measure
           of environment, then this distribution pz is consequentially conditioned on the environment
        '''
        #### PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS ####
        #NN 1/3
        self.Lambdaf_prior = nn.Sequential(
            nn.Linear(in_features = 5, out_features = 6),
            nn.ReLU(),
            nn.Linear(in_features = 6, out_features = 2*latent_features)
            )  #Output
        #### PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS ####
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        ##NN 2/3 (X,Y,E)-Encoder: Inference Network
        self.NonLinearEncoder = nn.Sequential(
            nn.Linear(in_features = 15, out_features = 6),
            nn.ReLU(),
            nn.Linear(in_features=6, out_features = 2*latent_features)
        )

        #For NN 3/3 (Decoder): Generative Model
        #Decode the latent sample `z` into the parameters of the observation model
        #`p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        self.NonLinearDecoder = nn.Sequential(
            nn.Linear(in_features = latent_features, out_features = 6),
            nn.ReLU(),
            nn.Linear(in_features = 6, out_features = 10),
        )

        

    def prior_params(self, y, e):
        """return the distribution `p(z)`"""
        ye = torch.cat((y, e), dim = 1)
        ye = ye.to(torch.float32)
        lambdaf_parameters = self.Lambdaf_prior(ye)

        return lambdaf_parameters

    def encoder(self, x, y, e):

        xye = torch.cat((x,y,e), dim = 1)

        xye = self.NonLinearEncoder(xye)
    
        return xye
   
    def decoder(self, z):
        
        x = self.NonLinearDecoder(z)
        return x    
        
    def posterior(self, x:Tensor, y:Tensor, z:Tensor) -> Distribution:
        """return the distribution `q(x|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        h_x = self.encoder(x, y, z)
        mu, log_sigma =  h_x.chunk(2, dim=-1)
        
        # return a distribution `q(x|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def prior(self, y, e)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params(y, e)

        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def observation_model(self, z:Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""
        px_means = self.decoder(z)
        px_means = px_means.view(-1, *self.input_shape)
        log_var = 0.01 * torch.ones(px_means.shape)
        log_sigma = torch.log(torch.sqrt(log_var))

        return ReparameterizedDiagonalGaussian(mu = px_means, log_sigma = log_sigma)
        

    def forward(self, x, y, e) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""

        #### Run through ENCODER and calculate mu and sigma for latent space sampling
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x, y, e)
        
        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        #### LATENT SPACE
        z = qz.rsample()
        
        #### DECODER
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        # define the prior p(z)
        pz = self.prior(y,e)
        
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}
    
    
    def sample_from_prior(self, y, e):
        """sample z~p(z) and return p(x|z)"""
        
        # define the prior p(z)
        pz = self.prior(y, e)
        
        # sample the prior 
        z = pz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'z': z}
    
    def reduce(self, x:Tensor) -> Tensor:
        """for each datapoint: sum over all dimensions"""
        return x.view(x.size(0), -1).sum(dim=1)

    def VariationalInference(self, x, y, e, beta):
        self.beta = beta
        # forward pass through the model
        outputs = self.forward(x, y, e)
        
        # unpack outputs
        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]
        
        # evaluate log probabilities
        log_px = self.reduce(px.log_prob(x)) #log(p(x|z))
        log_pz = self.reduce(pz.log_prob(z)) #log(p(z))
        log_qz = self.reduce(qz.log_prob(z)) #log(q(z|x))
        
        # compute the ELBO with and without the beta parameter: 
        # `L^\beta = E_q [ log p(x|z) ] - \beta * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        #########################################################################################################
        # Reconstruction loss: E_q [ log p(x|z) ]
        # Regularization term: \beta * D_KL(q(z|x) | p(z))`
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
