This is the implementation of Interaction Detection in PyTorch version. （Notice this implementation only includes the Interaction Detection part. The model compression part `ParaACE` is not included in this demo, please find it in the MxNet implementation folder.)

Three `ipynb` files are included.

`1-demo.ipynb` and `2-Inconsistency_NID.ipynb`  run for a few minutes. 
`3-drug_drug_interaction.ipynb`  is an implementation which is suitable for large-scale datasets. The dataset can be downloaded from https://zenodo.org/record/4135059 (around 420.1 MB zip file). But still you can check the printed results in the notebook, or run the code following the instruction.


`1-demo.ipynb` has four parts. 
1. We train the ReLU network what we need to analysis.
2. We show the Hessian given by analytical solution is a zero matrix.
3. We show the advantage proposed interaction measure.
4. We group the features according to the detected pairwise interactions.
In addition, we show the application of UCB algorithm to the detection of higher-order interactions. 

`2-Inconsistency_NID.ipynb` is for the prove of the inconsistency of NID method by a counterexample.
1. We run the NID code with a counterexample.
2. We compare the NID pairwise interaction results with ours.