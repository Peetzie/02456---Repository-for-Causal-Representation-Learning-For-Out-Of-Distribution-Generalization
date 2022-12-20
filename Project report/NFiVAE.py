#NF-iVAE class
import torch
from torch import nn, Tensor
from torch.distributions import Distribution
from functools import reduce
from torch.distributions import Normal
from typing import *

from ReparameterizedDiagonalGaussian import ReparameterizedDiagonalGaussian



def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)

class NFiVAE(nn.Module):
    # A nonfactorized identifiable Variational Autoencoder
    #The prior is conditioned on the environment, E, and target label Y through a Neural network and is non-factorised through training of both natural parameters 
    #and sufficient statistics.
    
    def __init__(self, input_shape:torch.Size, latent_features:int) -> None:
        super(NFiVAE, self).__init__()
        
        self.input_shape = input_shape
        self.latent_features = latent_features

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        '''
        According to page 31-32 of Lu et al the NFiVAE consist of 7 NNs:
        1. TNN prior
        2. lambdaNN prior
        3. lambdaf prior
        
        4. X-encoder (Classic image CNN)
        5. (Y, E)-encoder
        6. (X, Y, E)-merger/encoder

        7. Decoder

        1-3: Learn nonfactorised priors based on the label distribution for the given environment
        4-6: Encoding X, encoding Y and E and merging these two encoders, to generate a 
             qz which is conditional on the environment.
        7: Decodes the latent space through pz. Since the latent space now contain some measure
           of environment, then this distribution pz is consequentially conditioned on the environment
        '''
        #### PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS ####
        #NN 1/7
        self.TNN_prior = nn.Sequential(
            nn.Linear(in_features = latent_features, out_features=50), #Input
            nn.Linear(in_features = 50, out_features=50), #Fully connected
            nn.ReLU(),
            nn.Linear(in_features = 50, out_features = 45)) #Output

        #NN 2/7
        self.LambdaNN_prior = nn.Sequential(
            nn.Linear(in_features = 2, out_features=50), #Input
            nn.Linear(in_features = 50, out_features=50), #Fully connected
            nn.ReLU(),
            nn.Linear(in_features = 50, out_features = 45)) #Output

        #NN 3/7
        self.Lambdaf_prior = nn.Sequential(
            nn.Linear(in_features = 2, out_features=50), #Input
            nn.Linear(in_features = 50, out_features=50), #Fully connected
            nn.ReLU(),
            nn.Linear(in_features = 50, out_features = 20))  #Output
        #### PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS PRIORS ####

        #X-Encoder, NN 4/7
        self.encoderCNN1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        self.encoderCNN2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        self.encoderCNN3 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        ##(Y, E)-Encoder, NN 5/7
        self.YEencoder = nn.Sequential(
            nn.Linear(in_features = 2, out_features=100),
            nn.Linear(in_features = 100, out_features = 100),
            nn.ReLU())

        ##(X, Y, E)-merger/encoder, NN 6/7 
        self.XYEmerger = nn.Sequential(
            nn.Linear(in_features = 32 * 4 * 4 + 100, out_features=100),
            nn.Linear(in_features = 100, out_features = 100),
            nn.ReLU(),
            nn.Linear(in_features = 100, out_features = 2*latent_features))


        #(Decoder), NN 7/7
        self.decoderFFN = nn.Linear(in_features=latent_features, out_features = 32 * 4 * 4)
        self.decoderFFN2 = nn.Linear(in_features = 32 * 4 * 4, out_features = 32 * 4 * 4)
        
        self.decoderCNN1 = nn.ConvTranspose2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1
                              ,output_padding = 0)
        self.decoderCNN2 = nn.ConvTranspose2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1
                              ,output_padding = 1)
        self.decoderCNN3 = nn.ConvTranspose2d(in_channels = 32, out_channels = 3, kernel_size = 3, stride = 2, padding = 1
                              ,output_padding = 1)

    #Non-factorized prior for NF-iVAE from page 28 in the article by Lu et al.
    def non_factorized_prior_pz(self, z, TNN_parameters, lambdaNN_parameters, lambdaf_parameters, reduce = True):
        #Calculation from page 28 (appendix M.2) for P_T,lambda(Z|Y,E).
        z_cat = torch.cat((z, z.pow(2)), dim = 1) 
        # "(*).sum(dim=1)" is the vector-vector dot product. It is not possible to make this calculation with e.g. torch.tensordot, as we want the dot product of each vector and not the entire batch
        non_factorized_prior = (TNN_parameters*lambdaNN_parameters).sum(dim = 1) + (z_cat * lambdaf_parameters).sum(dim = 1)
        #the f,z term - (z_cat * lambdaf_parameters).sum(dim = 1): Is equivalent to the factorized exponential family
        #the NN term - (TNN_parameters*lambdaNN_parameters).sum(dim = 1): Capture the dependencies between the latent variables
        return non_factorized_prior

    def parameters_for_non_factorized_prior(self, z, y, e):
        #estimating the prior distribution parameters through the three neural networks 1-3/7 from above.
        #Correspond to the "prior params" definition in the other two scripts.
        TNN_parameters = self.TNN_prior(z)
        ye = torch.cat((y, e), dim = 1)
        ye = ye.to(torch.float32)
        LambdaNN_parameters = self.LambdaNN_prior(ye)
        lambdaf_parameters = self.Lambdaf_prior(ye)

        return TNN_parameters, LambdaNN_parameters, lambdaf_parameters

    def encoder(self, x, y, e):
        #Encoder
        #CNN X-encoder
        x = self.relu(self.encoderCNN1(x))
        x = self.relu(self.encoderCNN2(x))
        x = self.relu(self.encoderCNN3(x))
        x = x.view(x.size(0), -1) #NN 4/7

        ye = torch.cat((y, e), dim = 1)
        ye = ye.to(torch.float32)
        #Y,E-encoder
        ye = self.YEencoder(ye) #NN 5/7

        xye = torch.cat((x,ye), dim = 1)
        #Merger-encoder
        xye = self.XYEmerger(xye) #NN 6/7
    
        return xye
   
    def decoder(self, z):
        #Decoder NN 7/7
        x = self.relu(self.decoderFFN(z))
        x = self.relu(self.decoderFFN2(x))
        
        # reshape x and add CNN decoder
        x = x.view(-1, 32, 4, 4)
        
        x = self.relu(self.decoderCNN1(x))
        x = self.relu(self.decoderCNN2(x))
        x = self.decoderCNN3(x)
        return x
        
    def posterior(self, x:Tensor, y:Tensor, e:Tensor) -> Distribution:
        #Calculate the parameters in the distribution: q(z|x) ~ N()
        h_x = self.encoder(x, y, e)
        mu, log_sigma =  h_x.chunk(2, dim=-1)
        #Return a latent space probability distribution given the input image x.
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    
    def observation_model(self, z:Tensor) -> Distribution:
        #Calling the decoder
        px_loc = self.decoder(z)
        px_loc = px_loc.view(-1, *self.input_shape) # reshape the output #old
        shape = torch.ones(px_loc.shape)
        shape = shape.to(self.device)
        shape = 0.1 * shape 
        #Return the probability of observing our input X, given our constructed latent space p(X|Z)
        return Normal(loc=px_loc, scale = shape)
        

    def forward(self, x, y, e) -> Dict[str, Any]:
        #Forward pass through all steps of the VAE model.
        #### Run through ENCODER and calculate mu and sigma for latent space sampling

        # define the posterior q(z|x)
        qz = self.posterior(x, y, e)
        
        #### LATENT SPACE sampling using the reparameterization trick
        z = qz.rsample()
        
        #### DECODE
        px = self.observation_model(z)

        return {'px': px, 'qz': qz, 'z': z}

    def reduce(self, x:Tensor) -> Tensor:
        """for each datapoint: sum over all dimensions"""
        return x.view(x.size(0), -1).sum(dim=1)

    def VariationalInference(self, x, y, e):
        
        # Get the probability distributions px, pz, qz and the used latent space sample z.
        parameters_and_latent_space = self.forward(x, y, e)
        px_ze, qz_xye, z = [parameters_and_latent_space[k] for k in ["px", "qz", "z"]]


        # DEFINE THE PRIOR p(z)
        #### PRIOR
        z_temp = z.detach().requires_grad_(requires_grad = True)
        y_temp = y.detach()
        e_temp = e.detach()
        TNN_parameters, LambdaNN_parameters, lambdaf_parameters = self.parameters_for_non_factorized_prior(z_temp, y_temp, e_temp) #bruger temp da priors træning ikke skal påvirke encoder og decoder

        #prior calculation from page 28 (til differentiering ifm. eq. 64 skal dette ikke være logget, men når
        # elbo loss beregnes, skal man huske at tage log at denne værdi jvf. p. 28)

        #According to page 6. equation (8)-(10) and p. 28.
        TNN_parameters_hat = TNN_parameters.detach()
        LambdaNN_parameters_hat = LambdaNN_parameters.detach()
        lambdaf_parameters_hat = lambdaf_parameters.detach()

        #The non_factorized_prior_pz calculation is based on the equation in appendix M.2 on page 28 in Lu et al 2022. It is assumed that this equation show the log(prior).
        #Use detached prior parameters to only train the encoder/decoder through the elbo loss according to page 6 & 28 in Lu et al 2022
        log_pz_ye_ELBO =  self.non_factorized_prior_pz(z, TNN_parameters_hat, LambdaNN_parameters_hat, lambdaf_parameters_hat)

        #Use detached z to only train the prior parameters through the SM loss according to the description on page 6 & 28 in Lu et al
        log_pz_ye_SM = self.non_factorized_prior_pz(z_temp, TNN_parameters, LambdaNN_parameters, lambdaf_parameters) #phi_hat (or phi.detach()) from page. 6 is implemented implicitly/auto-
        #matically through the implementation of SM on page 28 eq. (64), where autograd of p_T,lambda is used instead of the conditional distribution q_phi(Z|X,Y,E), here phi
        #is evidently excluded from the gradient calculation.
        
        #### Score matching (SM) loss
        dpdz_ye = torch.autograd.grad(log_pz_ye_SM.sum(), z_temp, create_graph = True, retain_graph=True)[0]
        ddpdz_sq_ye = torch.autograd.grad(dpdz_ye.sum(), z_temp, create_graph = True, retain_graph=True)[0] 
        SM = (ddpdz_sq_ye + 0.5 * dpdz_ye.pow(2)).sum(1) #Calculation from page 28 equation (64)
        
        log_px_ze = self.reduce(px_ze.log_prob(x)) #log(p(x|z)): log probability of observing our input x, given our latent space - how good is the model recontruction?
        log_qz_xye = self.reduce(qz_xye.log_prob(z)) #log(q(z|x)): log probability of generating this latent space, given our posterior distribution
        
        #### ELBO loss
        #########################################################################################################
        # Reconstruction loss: E_q [ log p(x|z) ]
        # Kullback-leibler divergence: D_KL(q(z|x) | p(z))`
        #########################################################################################################
        kl = log_qz_xye - log_pz_ye_ELBO
        elbo = log_px_ze - kl
        
        # loss
        loss = -(elbo.mean() - SM.mean())

        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': elbo, 'log_px':log_px_ze, 'kl': kl}
        
        outputs = parameters_and_latent_space
        return loss, diagnostics, outputs
