#NF-iVAE class
import torch
from torch import nn, Tensor
from torch.distributions import Distribution
from functools import reduce
from torch.distributions import Normal
from typing import *

import ReparameterizedDiagonalGaussian



def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)

class NFiVAE(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """
    
    def __init__(self, input_shape:torch.Size, latent_features:int) -> None:
        super(NFiVAE, self).__init__()
        
        self.input_shape = input_shape
        self.latent_features = latent_features

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # define the parameters of the prior, chosen as p(z) = N(0, I)
        ## setting the prior to a vector consisting of zeros with dimensions (1,2*latent_features)
        # self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))
        
        '''
        According to page 31-32 the iVAE consist of 7 NNs:
        1. TNN prior
        2. lambdaNN prior
        3. lambdaf prior
        
        4. X-encoder (Classic image CNN)
        5. (Y, E)-encoder
        6. (X, Y, E)-merger/encoder

        7. Decoder

        1-3: Learn priors based on the label distribution for the given environment
        4-6: Encoding X, encoding Y and E and merging these two encoders, to generate a 
             qz which is conditional on the environment.
        7: Decodes the latent space through pz. Since the latent space now contain some measure
           of environment, then this distribution pz is consequentially conditioned on the environment

        NN 1-3 can be found in the variational inference funktion.
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

         #For NN 4/7 X-Encoder: Inference Network
         #Encode the observation `x` into the parameters of the posterior distribution
         #`q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoderCNN1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        self.encoderCNN2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        self.encoderCNN3 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        ##NN 5/7 (Y, E)-Encoder
        self.YEencoder = nn.Sequential(
            nn.Linear(in_features = 2, out_features=100),
            nn.Linear(in_features = 100, out_features = 100),
            nn.ReLU())

        ##NN 6/7 (X, Y, E)-merger/encoder
        #remember to concatenate x.flatten, y, e before running this.
        self.XYEmerger = nn.Sequential(
            nn.Linear(in_features = 32 * 4 * 4 + 100, out_features=100),
            nn.Linear(in_features = 100, out_features = 100),
            nn.ReLU(),
            nn.Linear(in_features = 100, out_features = 2*latent_features))


        #For NN 7/7 (Decoder): Generative Model
        #Decode the latent sample `z` into the parameters of the observation model
        #`p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        self.decoderFFN = nn.Linear(in_features=latent_features, out_features = 32 * 4 * 4)
        self.decoderFFN2 = nn.Linear(in_features = 32 * 4 * 4, out_features = 32 * 4 * 4)
        
        self.decoderCNN1 = nn.ConvTranspose2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1
                              ,output_padding = 0)
        self.decoderCNN2 = nn.ConvTranspose2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1
                              ,output_padding = 1)
        self.decoderCNN3 = nn.ConvTranspose2d(in_channels = 32, out_channels = 3, kernel_size = 3, stride = 2, padding = 1
                              ,output_padding = 1)

    #Non-factorized prior for NF-iVAE from page 28
    def non_factorized_prior_pz(self, z, TNN_parameters, lambdaNN_parameters, lambdaf_parameters, reduce = True):
        #beregning fra side 28 under M.2. P_T,lambda(Z|Y,E).
        z_cat = torch.cat((z, z.pow(2)), dim = 1) 
        # "(*).sum(dim=1)" is the vector-vector dot product. It is not possible to make this calculation with e.g. torch.tensordot, as we want the dot product of each vector and not the entire batch
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
        lambdaf_parameters = self.Lambdaf_prior(ye)

        return TNN_parameters, LambdaNN_parameters, lambdaf_parameters

    #NN 4/7
    def encoder(self, x, y, e):
        x = self.relu(self.encoderCNN1(x))
        x = self.relu(self.encoderCNN2(x))
        x = self.relu(self.encoderCNN3(x))
        x = x.view(x.size(0), -1) #NN 4/7

        ye = torch.cat((y, e), dim = 1)
        ye = ye.to(torch.float32)
        ye = self.YEencoder(ye) #NN 5/7

        xye = torch.cat((x,ye), dim = 1)
        xye = self.XYEmerger(xye) #NN 6/7
    
        return xye
   
    #NN 7/7
    def decoder(self, z):
        
        x = self.relu(self.decoderFFN(z))
        x = self.relu(self.decoderFFN2(x))
        
        # reshape x and add CNN decoder
        x = x.view(-1, 32, 4, 4)
        
        x = self.relu(self.decoderCNN1(x))
        x = self.relu(self.decoderCNN2(x))
        x = self.decoderCNN3(x)
        return x
        
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
        px_loc = px_loc.view(-1, *self.input_shape) # reshape the output #old
        shape = torch.ones(px_loc.shape)
        shape = shape.to(self.device)
        shape = 0.1 * shape 
        #sandsynlighedsfordeling der giver 1 eller 0, baseret på log-odds givet i logits input fra p(x|z).
        #Dvs. at px_logits angiver sandsynligheden for at det givne pixel er henholdsvist rød,grøn,blå. Pixel værdien
        #er enten 0 eller 1. Når man sampler fra bernoulli fordelingen fås dermed et billede, som givet z, giver en figur,
        #som er bestemt af de sandsynligheder der er i px_logits (p(x|z)). Dvs. at for et givet latents space, kan en
        #figur/et tal reproduceres ud fra de beregnede sandsynligheder og den efterfølgende sample fra Bernoulli fordelingen.
        #return Bernoulli(logits=px_loc, validate_args=False)
        return Normal(loc=px_loc, scale = shape)
        

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
        TNN_parameters, LambdaNN_parameters, lambdaf_parameters = self.parameters_for_non_factorized_prior(z_temp, y_temp, e_temp) #bruger temp da priors træning ikke skal påvirke encoder og decoder

        #prior calculation from page 28 (til differentiering ifm. eq. 64 skal dette ikke være logget, men når
        # elbo loss beregnes, skal man huske at tage log at denne værdi jvf. p. 28)

        #According to page 6. equation (8)-(10)
        TNN_parameters_hat = TNN_parameters.detach()
        LambdaNN_parameters_hat = LambdaNN_parameters.detach()
        lambdaf_parameters_hat = lambdaf_parameters.detach()

        log_pz_ye_ELBO =  self.non_factorized_prior_pz(z, TNN_parameters_hat, LambdaNN_parameters_hat, lambdaf_parameters_hat)
        log_pz_ye_SM = self.non_factorized_prior_pz(z_temp, TNN_parameters, LambdaNN_parameters, lambdaf_parameters) #phi_hat (or phi.detach()) from page. 6 is implemented implicitly/auto-
        #matically through the implementation of SM on page 28 eq. (64), where autograd of p_T,lambda is used instead of the conditional distribution q_phi(Z|X,Y,E), here phi
        #is evidently excluded from the gradient calculation.
        #OBS: HER SKAL DER IKKE ANVENDES LOG-PROBS AF PZ_YE JVF. SM DELEN AF EQ. (64) PÅ SIDE 28... eller hvad???
        dpdz_ye = torch.autograd.grad(log_pz_ye_SM.sum(), z_temp, create_graph = True, retain_graph=True)[0]

        #ddpz_ye = torch.autograd.grad(dpz_ye.mean(), z_temp, create_graph = True, retain_graph=True)[0] #changed sum() to mean()
        ddpdz_sq_ye = torch.autograd.grad(dpdz_ye.sum(), z_temp, create_graph = True, retain_graph=True)[0] #original

        #### SM loss SM loss SM loss SM loss SM loss SM loss SM loss SM loss ####
        #Calculation from page 28 equation 64
        SM = (ddpdz_sq_ye + 0.5 * dpdz_ye.pow(2)).sum(1)
        #### SM loss SM loss SM loss SM loss SM loss SM loss SM loss SM loss ####

        #### ELBO loss ELBO loss ELBO loss ELBO loss ELBO loss ELBO loss ELBO loss ####
        # evaluate log probabilities
        # Skal jvf. s. 32 ændres til at være en normal fordeling i stedet for en
        # bernoulli fordeling, og her skal mean være outputtet af decoderen og varians skal
        #blot sættes til at være 0.01
        
        log_px_ze = self.reduce(px_ze.log_prob(x)) #log(p(x|z)): Sandsynligheden for at observere vores input variabel x
        #givet vores latent space (tjekker modellens evne til at rekonstruere sig selv, ved at maximere sandsynlig-
        #heden for at observere inputtet selv, givet det konstruerede latent space.
        
        ####(old)log_pz = reduce(pz.log_prob(z)) #log(p(z)): 
        #log_pz_ye = torch.log(pz_ye)#Sandsynligheden for at observere vores latent space, givet at
        #latent space følger en standard-normal fordeling (Jo højere sandsynlighed jo bedre)
        
        log_qz_xye = self.reduce(qz_xye.log_prob(z)) #log(q(z|x)): Sandsynligheden for at generere netop dette latent space givet 
        #vores input billede x. Denne værdi skal helst være lav?
        
        # compute the ELBO: 
        # `L^\beta = E_q [ log p(x|z) ] - \beta * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        #########################################################################################################
        # Reconstruction loss: E_q [ log p(x|z) ]
        # Regularization term: \beta * D_KL(q(z|x) | p(z))` => Forsøger at tvinge fordelingen q(z|x) mod N(0,1)?
        #########################################################################################################
        
        kl = log_qz_xye - log_pz_ye_ELBO
        
        elbo = log_px_ze - kl
        ####
        
        # loss
        loss = -(elbo.mean() - SM.mean())
        #print("SM: {}. elbo: {} = log_px_ze: {} + log_pz_ye: {} - log_qz_xye: {}".format(SM.mean(), elbo.mean(),log_px_ze.mean(),pz_ye.mean(),log_qz_xye.mean()))

        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': elbo, 'log_px':log_px_ze, 'kl': kl}
        
        outputs = parameters_and_latent_space
        return loss, diagnostics, outputs

    #def sample_from_prior(self, batch_size:int=100):
    #    """sample z~p(z) and return p(x|z)"""
    #    
    #   # Laver bare reconstruction baseret på latent space
    #    #Kan evt. fjernes. Anvendes bare til at vise hvor god modellen er til at generere data baseret på
    #    #latent space genererede data. Funktionen anvendes kun i make_vae_plots.
    #    
    #    # degine the prior p(z)
    #    pz = self.prior(batch_size=batch_size)
    #    
    #    # sample the prior 
    #    z = pz.rsample()
    #    
    #    # define the observation model p(x|z) = B(x | g(z))
    #    px = self.observation_model(z)
    #    
    #    return {'px': px, 'pz': pz, 'z': z}
