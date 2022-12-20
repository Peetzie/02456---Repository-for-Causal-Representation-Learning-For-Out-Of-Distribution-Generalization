#NF-iVAE class for synthetic data
import torch
from torch import nn, Tensor
from torch.distributions import Distribution
from functools import reduce
from torch.distributions import Normal
from typing import *
from ReparameterizedDiagonalGaussian import ReparameterizedDiagonalGaussian


class NFiVAE_synth(nn.Module):

    
    def __init__(self, input_shape:torch.Size, latent_features:int) -> None:
        super(NFiVAE_synth, self).__init__()
        
        self.input_shape = input_shape
        self.latent_features = latent_features
        
        '''
        According to page 31-32 the NF-iVAE consist of 5 NNs:
        1. TNN prior
        2. lambdaNN prior
        3. lambdaf prior
        
        4. (X, Y, E)-merger/encoder

        5. Decoder

        1-3: Learn priors based on the label distribution for the given environment
        4: Encoding X, Y and E merged together, to generate a 
             qz which is conditional on the environment.
        5: Decodes the latent space through pz. Since the latent space now contain some measure
           of environment, then this distribution pz is consequentially conditioned on the environment

        NN 1-3 can be found in the variational inference funktion.
        '''

        #1/7 - lambdaf prior
        self.Lambdafnl_prior = nn.Sequential(
            nn.Linear(in_features = 5, out_features = 6),
            nn.ReLU(),
            nn.Linear(in_features = 6, out_features = 2*latent_features)
        )

        #2/7 - TNN prior
        self.TNN_prior = nn.Sequential(
            nn.Linear(in_features = latent_features, out_features = 6),
            nn.ReLU(),
            nn.Linear(in_features = 6, out_features = latent_features)
        )

        #3/7 - LambdaNN prior
        self.LambdaNN_prior = nn.Sequential(
            nn.Linear(in_features = 5, out_features = 6),
            nn.ReLU(),
            nn.Linear(in_features = 6, out_features = 1)
        )

        #4/5 - Merger/encoder
        self.NonLinearEncoder = nn.Sequential(
            nn.Linear(in_features = 15, out_features = 6),
            nn.ReLU(),
            nn.Linear(in_features=6, out_features = 2*latent_features)
        )

        #5/5 Decoder
        self.NonLinearDecoder = nn.Sequential(
            nn.Linear(in_features = latent_features, out_features = 6),
            nn.ReLU(),
            nn.Linear(in_features = 6, out_features = 10),
        )

        self.relu = nn.ReLU()
        

    #Equation (2) from paper
    def prior_pdf(self, z, TNN_parameters, lambdaNN_parameters, lambdafnl_parameters, reduce = True):
        nn = (TNN_parameters*lambdaNN_parameters).sum(dim = 1)
        z_cat = torch.cat((z, z.pow(2)), dim = 1) 
        f = (z_cat * lambdafnl_parameters).sum(dim = 1)
        return nn + f

    def prior_params(self, z, y, e):
        """return the distribution `p(z)`"""
        TNN_parameters = self.TNN_prior(z)
        ye = torch.cat((y, e), dim = 1)
        ye = ye.to(torch.float32)
        LambdaNN_parameters = self.LambdaNN_prior(ye)
        lambdafnl_parameters = self.Lambdafnl_prior(ye)

        return TNN_parameters, LambdaNN_parameters, lambdafnl_parameters

    def encoder(self, x, y, e):
        xye = torch.cat((x,y,e), dim = 1)

        xye = self.NonLinearEncoder(xye)
        
        return xye
   
    def decoder(self, z):
        x = self.NonLinearDecoder(z)
        return x

     #Non-factorized prior for NF-iVAE from page 28
    def non_factorized_prior_pz(self, z, TNN_parameters, lambdaNN_parameters, lambdaf_parameters, reduce = True):
        z_cat = torch.cat((z, z.pow(2)), dim = 1) 

        non_factorized_prior = (TNN_parameters*lambdaNN_parameters).sum(dim = 1) + (z_cat * lambdaf_parameters).sum(dim = 1)
        #the f,z term - (z_cat * lambdaf_parameters).sum(dim = 1): Is equivalent to the factorized exponential family
        #the NN term - (TNN_parameters*lambdaNN_parameters).sum(dim = 1): Capture the dependencies between the latent variables
        return non_factorized_prior

    def parameters_for_non_factorized_prior(self, z, y, e):
        """return the distribution `p(z)`"""
        TNN_parameters = self.TNN_prior(z)
        ye = torch.cat((y, e), dim = 1)
        ye = ye.to(torch.float32)
        LambdaNN_parameters = self.LambdaNN_prior(ye)
        lambdaf_parameters = self.Lambdafnl_prior(ye)

        return TNN_parameters, LambdaNN_parameters, lambdaf_parameters
        
    def posterior(self, x:Tensor, y:Tensor, e:Tensor) -> Distribution:
        """return the distribution `q(x|x) = N(z | \mu(x), \sigma(x))`"""
        # compute the parameters of the posterior
        h_x = self.encoder(x, y, e)
        mu, log_sigma =  h_x.chunk(2, dim=-1)
        
        # return a distribution `q(x|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    
    def observation_model(self, z:Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""
        px_loc = self.decoder(z)
        shape = torch.ones(px_loc.shape)
        shape = 0.1 * shape
        return Normal(loc=px_loc, scale = shape)
        

    def forward(self, x, y, e) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""
        #### Run through ENCODER and calculate mu and sigma for latent space sampling
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x, y, e)
        
        #### LATENT SPACE
        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()
        
        #### DECODER
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)

        return {'px': px, 'qz': qz, 'z': z}

    def reduce(self, x:Tensor) -> Tensor:
        """for each datapoint: sum over all dimensions"""
        return x.view(x.size(0), -1).sum(dim=1)

    def VariationalInference(self, x, y, e):
        # forward pass through the model to get the encoder and decoder outputs
        parameters_and_latent_space = self.forward(x, y, e)

        # unpack encoder parameters from (px), decoder parameters (qz) and the latent space (z)
        px_ze, qz_xye, z = [parameters_and_latent_space[k] for k in ["px", "qz", "z"]]

        # DEFINE THE PRIOR p(z)
        #### PRIOR
        z_temp = z.detach().requires_grad_(requires_grad = True)
        y_temp = y.detach()
        e_temp = e.detach()
        #prior calculation from page 28
        #Uses detach, as training of priors should not affect the encoder and decoder
        TNN_parameters, LambdaNN_parameters, lambdaf_parameters = self.parameters_for_non_factorized_prior(z_temp, y_temp, e_temp)

        #According to page 6. equation (8)-(10)
        TNN_parameters_hat = TNN_parameters.detach()
        LambdaNN_parameters_hat = LambdaNN_parameters.detach()
        lambdaf_parameters_hat = lambdaf_parameters.detach()

        log_pz_ye_ELBO =  self.non_factorized_prior_pz(z, TNN_parameters_hat, LambdaNN_parameters_hat, lambdaf_parameters_hat)
        log_pz_ye_SM = self.non_factorized_prior_pz(z_temp, TNN_parameters, LambdaNN_parameters, lambdaf_parameters)

        dpdz_ye = torch.autograd.grad(log_pz_ye_SM.sum(), z_temp, create_graph = True, retain_graph=True)[0]

        ddpdz_sq_ye = torch.autograd.grad(dpdz_ye.sum(), z_temp, create_graph = True, retain_graph=True)[0]

        #### SM loss SM loss SM loss SM loss SM loss SM loss SM loss SM loss ####
        #Calculation from page 28 equation 64
        SM = (ddpdz_sq_ye + 0.5 * dpdz_ye.pow(2)).sum(1)
        #### SM loss SM loss SM loss SM loss SM loss SM loss SM loss SM loss ####

        #### ELBO loss ELBO loss ELBO loss ELBO loss ELBO loss ELBO loss ELBO loss ####
        # evaluate log probabilities
        log_px_ze = self.reduce(px_ze.log_prob(x)) #log(p(x|z))
        log_qz_xye = self.reduce(qz_xye.log_prob(z)) #log(q(z|x))
        kl = log_qz_xye - log_pz_ye_ELBO
        elbo = log_px_ze - kl
        #### ELBO loss ELBO loss ELBO loss ELBO loss ELBO loss ELBO loss ELBO loss ####
        
        # loss
        loss = -(elbo.mean() - SM.mean())

        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': elbo, 'log_px':log_px_ze, 'kl': kl}
        
        outputs = parameters_and_latent_space
        return loss, diagnostics, outputs
