## One may use the following additive kernel to impose the interaction knowledge in GP model. 
## The following code component can be inserted into Botorch examples.           

from gpytorch.kernels import RBFKernel
add_kernel  = RBFKernel(active_dims=torch.tensor([0])) + \
RBFKernel(active_dims=torch.tensor([1]))+  \
RBFKernel(active_dims=torch.tensor([2]))+ \
RBFKernel(active_dims=torch.tensor([3])) + \
RBFKernel(active_dims=torch.tensor([4]))+  \
RBFKernel(active_dims=torch.tensor([5]))+ \
RBFKernel(active_dims=torch.tensor([6]))+  \
RBFKernel(active_dims=torch.tensor([7])) + \
RBFKernel(active_dims=torch.tensor([5,7]))+ RBFKernel(active_dims=torch.tensor([0,7]))+RBFKernel(active_dims=torch.tensor([5,0]))
                  
   

models = []
for i in range(train_y.shape[-1]):
    models.append(
        SingleTaskGP(
            train_x, train_y[..., i : i + 1], outcome_transform=Standardize(m=1), covar_module =add_kernel
        )
    )
model = ModelListGP(*models)
mll = SumMarginalLogLikelihood(model.likelihood, model)
