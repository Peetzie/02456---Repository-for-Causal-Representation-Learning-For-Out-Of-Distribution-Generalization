#NF-iVAE class for synthetic data
import torch
from torch import nn, Tensor
from torch.distributions import Distribution
from functools import reduce
from torch.distributions import Normal
from typing import *
from ReparameterizedDiagonalGaussian import ReparameterizedDiagonalGaussian


class NFiVAE_synth(nn.Module):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_\theta(x | z) = B(x | g_\theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """
    
    def __init__(self, input_shape:torch.Size, latent_features:int) -> None:
        super(NFiVAE_synth, self).__init__()
        
        self.input_shape = input_shape
        self.latent_features = latent_features

        
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
        ####
        #1/8
        # self.Lambdaflinear_prior = nn.Sequential(
        #     nn.Linear(in_features = 2, out_features = 50),
        #     nn.Linear(in_features = 50, out_features = 4)
        # )


        #2/8
        self.Lambdafnl_prior = nn.Sequential(
            nn.Linear(in_features = 5, out_features = 6),
            # nn.Linear(in_features = 5, out_features = 128),
            # nn.Linear(in_features = 128, out_features = 6),
            nn.ReLU(),
            nn.Linear(in_features = 6, out_features = 2*latent_features)
        )

        #3/8
        self.TNN_prior = nn.Sequential(
            nn.Linear(in_features = latent_features, out_features = 6),
            # nn.Linear(in_features = 2, out_features = 128),
            # nn.Linear(in_features = 128, out_features = 6),
            nn.ReLU(),
            nn.Linear(in_features = 6, out_features = latent_features)
        )

        #4/8
        self.LambdaNN_prior = nn.Sequential(
            nn.Linear(in_features = 5, out_features = 6),
            # nn.Linear(in_features = 5, out_features = 128),
            # nn.Linear(in_features = 128, out_features = 6),
            nn.ReLU(),
            nn.Linear(in_features = 6, out_features = 1)
        )

        #5/8
        # self.LinearEncoder = nn.Sequential(
        #     nn.Linear(in_features = latent_features, out_features = 50),
        #     nn.Linear(in_features = 50, out_features = 3)
        # )

        #6/8
        self.NonLinearEncoder = nn.Sequential(
            nn.Linear(in_features = 15, out_features = 6),
            # nn.Linear(in_features = 15, out_features = 128),
            # nn.Linear(in_features = 128, out_features = 6),
            nn.ReLU(),
            nn.Linear(in_features=6, out_features = 2*latent_features)
        )

        #7/8
        # self.LinearDecoder = nn.Sequential(
        #     nn.Linear(in_features = latent_features, out_features = 50),
        #     nn.Linear(in_features = 50, out_features = 2),
        # )

        #8/8
        self.NonLinearDecoder = nn.Sequential(
            nn.Linear(in_features = latent_features, out_features = 6),
            # nn.Linear(in_features = 2, out_features = 128),
            # nn.Linear(in_features = 128, out_features = 6),
            nn.ReLU(),
            nn.Linear(in_features = 6, out_features = 10),
        )

        self.relu = nn.ReLU()
        

    #Funktion fra Thea, ved ikke helt hvor hun har den fra - ligner dog equation (2)
    def prior_pdf(self, z, TNN_parameters, lambdaNN_parameters, lambdafnl_parameters, reduce = True):
        #beregning fra side 28 under M.2. P_T,lambda(Z|Y,E).
        nn = (TNN_parameters*lambdaNN_parameters).sum(dim = 1) #prikprodukt
        z_cat = torch.cat((z, z.pow(2)), dim = 1) 
        f = (z_cat * lambdafnl_parameters).sum(dim = 1) #prik produkt
        return nn + f

    def prior_params(self, z, y, e):
        """return the distribution `p(z)`"""
        TNN_parameters = self.TNN_prior(z)
        ye = torch.cat((y, e), dim = 1)
        ye = ye.to(torch.float32)
        LambdaNN_parameters = self.LambdaNN_prior(ye)
        lambdafnl_parameters = self.Lambdafnl_prior(ye)

        return TNN_parameters, LambdaNN_parameters, lambdafnl_parameters

    #NN 4/7
    def encoder(self, x, y, e):

        #x = relu(self.encoderCNN1(x))
        #x = relu(self.encoderCNN2(x))
        #x = relu(self.encoderCNN3(x))
        #x = x.view(x.size(0), -1) #NN 4/7
        
        #ye = torch.cat((y, e), dim = 1)
        # ye = ye.to(torch.float32)
        # ye = self.YEencoder(ye) #NN 5/7

        xye = torch.cat((x,y,e), dim = 1)
        #print(xye.size())
        xye = self.NonLinearEncoder(xye)
        # xye = self.XYEmerger(xye) #NN 6/7
        #mu, log_sigma =  x.chunk(2, dim=-1)
        
        return xye
   
    #NN 7/7
    def decoder(self, z):
        x = self.NonLinearDecoder(z)
        return x

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
        #print(z.size())
        px_loc = self.decoder(z)
        shape = torch.ones(px_loc.shape)
        shape = 0.1 * shape 
        #print(px_loc.size())
        #px_loc = px_loc.view(-1, *self.input_shape) # reshape the output #old
        #sandsynlighedsfordeling der giver 1 eller 0, baseret p� log-odds givet i logits input fra p(x|z).
        #Dvs. at px_logits angiver sandsynligheden for at det givne pixel er henholdsvist r�d,gr�n,bl�. Pixel v�rdien
        #er enten 0 eller 1. N�r man sampler fra bernoulli fordelingen f�s dermed et billede, som givet z, giver en figur,
        #som er bestemt af de sandsynligheder der er i px_logits (p(x|z)). Dvs. at for et givet latents space, kan en
        #figur/et tal reproduceres ud fra de beregnede sandsynligheder og den efterf�lgende sample fra Bernoulli fordelingen.
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
