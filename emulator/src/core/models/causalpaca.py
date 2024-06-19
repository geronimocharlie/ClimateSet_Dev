
import torch
from torch import Tensor
import torch.nn as nn
from emulator.src.core.models.basemodel import BaseModel
from emulator.src.core.models.model_utils import Gumbel_Sigmoid, ALM, TrExpScipy, init_weigths, WeightClipper
import collections
import hydra
from omegaconf import DictConfig
from typing import Optional, Union, List

class Causalpaca(BaseModel):
    def __init__(
        self, in_var_ids: List[str], 
        out_var_ids: List[str], 
        latent_size: int, 
        lon: int, 
        lat: int, 
        tau: int, 
        seq_len: int, 
        base_model_config: DictConfig, 
        datamodule_config: Optional[DictConfig] = None, 
        #sparsity_coff: float = 0.4, 
        channels_last: bool = False, 
        instantaneous: bool = True,
        hard_gumbel: bool = False,
        ortho_omega_gamma: float = 0.01,
        ortho_omega_mu: float = 0.9,
        ortho_mu_init: float = 1e-8,
        ortho_mu_mult_factor: int = 2,
        ortho_h_threshold: float = 1e-4,
        ortho_min_iter_convergence: float = 100,
        sparsity_omega_gamma: float = 0.01,
        sparsity_omega_mu: float = 0.95,
        sparsity_mu_init: int = 4,
        sparsity_mu_mult_factor: float = 1.2,
        sparsity_h_threshold: float = 1e-4,
        sparsity_min_iter_convergence: int = 1000,    
        acyclic_omega_gamma: float = 0.01,
        acyclic_omega_mu: float = 0.9,
        acyclic_mu_init: int = 1,
        acyclic_mu_mult_factor: int = 2,
        acyclic_h_threshold: float = 1e-8,
        acyclic_min_iter_convergence: int = 1000,
        use_grad_projection: bool = True,
        *args, **kwargs
    ):
        
        super().__init__(datamodule_config=datamodule_config, *args, **kwargs)
        print("in init")
        if datamodule_config is not None:
            if datamodule_config.get("channels_last") is not None:
                self.channels_last = datamodule_config.get("channels_last")
            if datamodule_config.get("lon") is not None:
                self.lon = datamodule_config.get("lon")
            if datamodule_config.get("lat") is not None:
                self.lat = datamodule_config.get("lat")
            if datamodule_config.get("seq_len") is not None:
                self.seq_len = datamodule_config.get("seq_len")
        else:
            self.lon = lon
            self.lat = lat
            self.channels_last = channels_last
            self.seq_len = seq_len
        
        self.save_hyperparameters()

        self.num_input_vars = len(in_var_ids)
        self.num_output_vars = len(out_var_ids) # determines number of causal heads
        self.tau = tau
        print("num input vars", self.num_input_vars)
        print("num output vars", self.num_output_vars)
        print("seq len", self.seq_len)
        print("tao", self.tau)
        assert (self.seq_len%self.tau)==0, f"Full sequence must be slicable by tau {self.tau} without remainders!"
        # we slice the full sequence in time windows of size tau
        # num time heads = self.seq_len/self.tau
        self.num_time_heads = self.seq_len//self.tau
        print("number time heads", self.num_time_heads)
    
        # we need a set of basemodels... (sequence to sequence)
        # TODO: update base_model
        # TODO: update base model to output second vector for long-term dependency
        self.base_model = torch.nn.Linear(self.num_input_vars * self.lon * self.lat * self.seq_len, self.num_output_vars * latent_size * self.seq_len)#TODO how to instantiate from here? without circulare dependencies and model configs bla?
    

        self.latent_size=latent_size # assumping same latent size for ys and xs could be changed though
        #self.sparsity_coff=sparsity_coff
        self.tau = tau 
        self.heads = {}
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.channels_last=channels_last

        self.gumbel_sigmoid = Gumbel_Sigmoid() # TODO: check hparamsarams / pass on hparamsarams for that

        # one head per output var
        for head_var in range(self.num_output_vars):
            # one head per time window
            self.heads[head_var]={}
          
            for head_time in range(self.num_time_heads):
                self.heads[head_var][head_time]={}
            
                # one transition matrix for each time step 
                for j in range(self.tau): 
                    #self.heads[head_var][head_time][j]=nn.ModuleList()
                    # Gs: latent_size -> latent_size for each time step up to tow!
                    #| BINARY WEIGHTS + sparsity constraint -> for now, sigmoid gumble + sparsity penalty
                    G_y=torch.nn.Linear(latent_size, latent_size, bias=False).apply(init_weigths) # initialize with 5s
                    
                    #| BINARY WEIGHTS + sparsity constraint -> for now, sigmoid gumble + sparsity penalty
                    G_z=torch.nn.Linear(latent_size, latent_size, bias=False).apply(init_weigths) # iniitalize with 5s
                    
                    # F: latent_size -> lon x lat (one variable at each grid point)
                    # normal layer + orthoganility & acyclicity constraint
                    F=torch.nn.Linear(latent_size, self.lon * self.lat, bias=False).apply(init_weigths) # init with 5s
                    
                    head = torch.nn.Sequential(
                        collections.OrderedDict(
                            [
                                (f"c_G_y_head_var_{head_var}_time_{head_time}_tau_{j}", G_y),
                                (f"c_Gumble_Sigmoid_y_head_var_{head_var}_time_{head_time}_tau_{j}", self.gumbel_sigmoid),
                                (f"c_G_z_head_var_{head_var}_time_{head_time}_tau_{j}", G_z),
                                (f"c_Gumble_Sigmoid_z_head_var_{head_var}_time_{head_time}_tau_{j}", self.gumbel_sigmoid),
                                (f"c_F_head_head_var_{head_var}_time_{head_time}_tau_{j}", F),
                                (f"c_F_Sigmoid_head_var_{head_var}_time_tau_{j}", self.gumbel_sigmoid),     
                            ]
                            )
                        )

                    self.heads[head_var][head_time][j]=head

        print(self.heads)
        #self.heads.to(device)
        #self.base_model.to(device)


        # crating ALMs : one per constraint
        # initialize ALM/QPM for orthogonality and acyclicity constraints
        self.ALM_ortho = ALM(self.hparams.ortho_mu_init,
                             self.hparams.ortho_mu_mult_factor,
                             self.hparams.ortho_omega_gamma,
                             self.hparams.ortho_omega_mu,
                             self.hparams.ortho_h_threshold,
                             self.hparams.ortho_min_iter_convergence,
                             dim_gamma=(self.latent_size, self.latent_size))
        if instantaneous:
            # add the acyclicity constraint if the instantaneous connections
            # are considered
            self.QPM_acyclic = ALM(self.hparams.acyclic_mu_init,
                                   self.hparams.acyclic_mu_mult_factor,
                                   self.hparams.acyclic_omega_gamma,
                                   self.hparams.acyclic_omega_mu,
                                   self.hparams.acyclic_h_threshold,
                                   self.hparams.acyclic_min_iter_convergence)
        
        self.ALM_sparsity = ALM(self.hparams.sparsity_mu_init,
                             self.hparams.sparsity_mu_mult_factor,
                             self.hparams.sparsity_omega_gamma,
                             self.hparams.sparsity_omega_mu,
                             self.hparams.sparsity_h_threshold,
                             self.hparams.sparsity_min_iter_convergence)

        # clipping weights to positive
        if self.use_grad_projection:
            self.weight_clipper = WeightClipper(min=0)

    def get_adj(self):
        """
        Returns: Matrices of the probabilities from which the masks linking the
        latent variables are sampled
        
        Returns dict with G_z, G_y, F for each head (target variable x time window head)

        shapes: 
            G: tau x latent_size x laent_size
            F: tax x latent_size x (lan x lot)

        """
        adj={}

        # for each target var
        for head_var in range(self.num_output_vars):
            adj[head_var]={}
            # for each time window head
            for head_time in range(self.num_time_heads):
                adj[head_var][head_time]={}
            
                for aspect in ["G_z", "G_y", "F"]:
                    weigths=[]
                    # one G per time step
                    for tau in range(self.tau):
                        # stack Gs along first axis to obtain (tau x latent_size x latent_size)
                        weigths.append([param.data for name, param in self.heads[head_var][head_time][tau].named_parameters() if (aspect in name)][0])
                    weights = torch.stack(weigths)
                    # swap axis bc of torch weight encoding
                    weigths = torch.swapaxes(weights, 1, 2)
                    adj[head_var][head_time].update({aspect:weigths})
              
        return adj
       
    
    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        # creaty empty tensor of output shape
        # batch size x seq len x vars x lon x lat
        print("batch size in forward", x.size())
        if self.channels_last:
              y = torch.empty(
            x.shape[0], x.shape[1], x.shape[-3], x.shape[-2], self.num_output_vars
            )#.to(device)
        else:
            y = torch.empty(
              
                x.shape[0], x.shape[1], self.num_output_vars, x.shape[-2], x.shape[-1]
            )#.to(device)
        #print("target shape", y.size())
        # pass full x through base model
        # TODO: remove flatten for real base model
        base_model_in = x.view(x.shape[0], -1)
        #print("base in", base_model_in)
        z = self.base_model(base_model_in) # batch size x (seq_len x latent_size) x num_output_vars
        #print("zs", z.size())
        if self.channels_last:
            z=z.view(x.shape[0], self.num_time_heads, self.tau, self.latent_size, self.num_output_vars) # (batch_size x num_windows x tau x latent_size) x num_output_vars)
        else:
            z=z.view(x.shape[0], self.num_output_vars, self.num_time_heads, self.tau, self.latent_size) # (batch_size x num_output_vars x num_windows x tau x latent_size))
        
        #print("z reshape", z.size())
        # fill in the values per target variable
        for i in range(self.num_output_vars):
            for ht in range(self.num_time_heads):
                for t in range(self.tau):
                    # pass z through respective head
                    # reshape
                    if self.channels_last:
                        out = self.heads[i][ht][t](z[:, ht, t, :, i]).view(-1, 1, self.lon, self.lat, 1)
                        y[:, (ht+t):(ht+t+1), :, :, i:(i+1)] = out
                    else:
                        out = self.heads[i][ht][t](z[:, i, ht, t, :]).view(-1, 1, 1, self.lon, self.lat)
                        y[:, (ht+t):(ht+t+1), i:(i+1), :, :] = out
                
        return y
    

    def add_additional_losses(self, loss: Tensor, epoch: int):

        # TODO schedule adding according to opoch
        adjacencies = self.get_adj()


        # for each head collect list of violation scores per constraint
        penalties_sparsity_z=[]
        penalties_sparsity_y=[]
        penalties_acyc_z=[] #TODO: only if not instantaneous connections
        penalties_acyc_y=[] #TODO: only if not instantaneous connections
        penalties_ortho=[]

        for head in range(self.num_output_vars):

            for head_t in range(self.num_time_heads):
                # compute penalties for all aspetcs
                #penalties_acyc_z.append(self.get_acyclicity_violation(adjacencies[head][head_t]["G_z"], epoch=epoch))
                #penalties_acyc_y.append(self.get_acyclicity_violation(adjacencies[head][head_t]["G_y"], epoch=epoch))

                penalties_ortho.append(self.get_ortho_violation(adjacencies[head][head_t]["F"], epoch=epoch))
                
                penalties_sparsity_y.append(self.get_sparsity_violation(adjacencies[head][head_t]["G_y"], epoch=epoch))
                penalties_sparsity_z.append(self.get_sparsity_violation(adjacencies[head][head_t]["G_z"], epoch=epoch))

        # compute average
        #h_acyc_z = torch.mean(penalties_acyc_z)
        #h_acyc_y = torch.mean(penalties_acyc_y)

        h_sparsity_z = torch.mean(penalties_sparsity_z)
        h_sparsity_y = torch.mean(penalties_sparsity_y)
        h_ortho = torch.mean(penalties_ortho)
        
        # add using ALMS (check convergence criteria)

        new_loss = loss
        loss_components_dict = {}

        #loss_components_dict["h_acyc_z"]=penalties_acyc_z
        #loss_components_dict["h_acyc_y"]=penalties_acyc_y
        loss_components_dict["h_sparsity_z"]=penalties_sparsity_z
        loss_components_dict["h_sparsity_z"]=penalties_sparsity_y
        loss_components_dict["h_ortho"]=penalties_ortho

        print(loss_components_dict)


        return new_loss, loss_components_dict

    def perform_weight_regularization(self):
        # performed in train step by base model, overwrite to perform clamping of weights (>0)

        if self.use_grad_project:
            with torch.no_grad():
                for head_var in range(self.num_output_vars):
                    for head_time in range(self.num_time_heads):
                        for j in range(self.tau):
                            # adjust to perform on all Gs and Fs!
                            self.heads[head_var][head_time][j].apply(self.weight_clipper)
                            assert torch.min(self.heads[head_var][head_time][j].weight.data) >= 0.

    
    def get_acyclicity_violation(self, w: Tensor, epoch: int) -> torch.Tensor:
        
        if epoch>0:
            h = self.compute_dag_constraint(w) / self.hparams.acyclic_constraint_normalization
        else:
            h= torch.tensor([0.])

        return h

    def get_ortho_violation(self, w: torch.Tensor, epoch: int) -> float:
       
        # scheduling?
        if epoch > self.hparams.schedule_ortho:
            # constraint = torch.tensor([0.])
            k = w.size(2)
            # for i in range(w.size(0)):
            #     constraint = constraint + torch.norm(w[i].T @ w[i] - torch.eye(k), p=2)
            i = 0
            # constraint = torch.norm(w[i].T @ w[i] - torch.eye(k), p=2, dim=1)
            constraint = w[i].T @ w[i] - torch.eye(k)
            h = constraint / self.ortho_normalization
        else:
            h = torch.tensor([0.])
        return h

   
    
    def get_sparsity_violation(self, w: Tensor, epoch: int, lower_threshold: float = 0.1, upper_threshold: float= 0.3) -> float:
        """
        To
        Calculate the number of causal links in the adjacency matrix, and constrain this to be less than a certain number.
        Threshold is the fraction of causal links, e.g. 0.1, 0.3
        """
        if epoch > self.hparams.schedule_sparsity:

            # NOTE:(seb) try p=2
            sum_of_connections = torch.norm(w, p=1) / self.sparsity_normalization
            #print('constraint value, before I subtract a threshold from it:', sum_of_connections)
            
           # If the sum_of_connections is greater than the upper threshold, then we have a violation
            if sum_of_connections > upper_threshold:
                constraint = sum_of_connections - upper_threshold
            
            # If the constraint is less than the lower threshold, then we also have a violation
            elif sum_of_connections < lower_threshold:
                constraint = lower_threshold - sum_of_connections

            # Otherwise, there is no penalty due to the constraint:
            else:
                constraint = torch.tensor([0.])
            
            #print('constraint value, after I subtract a threshold, or whatever:', constraint)

            #NOTE:(seb) - here we implement an inequality constraint rather than a forced equality
            h = torch.max(constraint, torch.tensor([0.]))

            

        else:
            h = torch.tensor([0.])

        #print('***SPARSITY VIOLATION*** INSIDE GET_SPARSITY_VIOLATION:', h)

        assert torch.is_tensor(h)
    
        return h

    def compute_dag_constraint(self, w_adj):
        """
        Compute the DAG constraint of w_adj
        :param np.ndarray w_adj: the weighted adjacency matrix (each entry in [0,1])
        """
        print(w_adj)
        assert (w_adj >= 0).detach().cpu().numpy().all()
        h = TrExpScipy.apply(w_adj) - w_adj.shape[0]
        return h


if __name__ == "__main__":

    seq_len=16
    tau=8
    lon=32
    lat=32
    in_var_ids=["BC", "CO2"]
    out_var_ids=["pr", "tas"]
    n_in=len(in_var_ids)
    n_out=len(out_var_ids)
    batch_size=8
    base_model_config = {'name': 'climax'}
    channels_last=True
    md = Causalpaca(
        in_var_ids=in_var_ids,
        out_var_ids=out_var_ids,
        lon=lon,
        lat=lat, 
        seq_len=seq_len,
        tau=tau,
        latent_size=10, 
        channels_last=channels_last,
        base_model_config=base_model_config

    )
    if channels_last:
        x = torch.ones((batch_size, seq_len, lon, lat, n_in))
    else:
        x = torch.ones((batch_size, seq_len, n_in, lon, lat))
    
    adj = md.get_adj()
    print(adj.keys())
    print(adj[0].keys())
    print(adj[0][0].keys())
    print("G", adj[0][0]["G_z"].size())
    print("F", adj[0][0]["F"].size())

    y = md.forward(x)

    l, d = md.add_additional_losses(0,1)
    print(y.shape)
