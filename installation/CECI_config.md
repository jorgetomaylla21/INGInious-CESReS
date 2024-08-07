# Configuration of the CECI for running the CESReS model

Author: Guillaume Steveny; 
Year: 2023 - 2024

To launch the model, one can use the CECI clusters to benefit from high performing components.
However, creating the complete infrastructure needed on these clusters is not a trivial task.
Training the model on your personal computer is of no solution if you have not enough ressources to load the datasets.

In this file, you will find a brief description on how the CECI could be used to run the CESReS model pipeline.
At the time of writing, the only cluster on which the environment was deployed is the `dragon2` cluster.
For other computing nodes, you will have to load other modules and perform other operations we do not explain here.
The way you could connect to this cluster instance will not be explained in detail here, but you can follow the 
steps indicated in this online tutorial: https://support.ceci-hpc.be/doc/_contents/QuickStart/ConnectingToTheClusters/index.html.

Once connected to the infrastructure, you should transfer the programs and components of the CESReS system.
This can be achieved with `scp`, `sftp` or simply by cloning the GitHub repository in your `$HOME` directory.

For explanation simplicity, this document will suppose you stored all the components inside a directory named `$HOME/cesres`.

However, once done, you should not run directly the program in the terminal you had when connecting to the clusters.
Each computation job should be submitted through a job manager which allocate the needed ressources.
Furthermore, python does not exist natively on this terminal. 
You should load modules to get access to some packages, libraries and programs.
The following section will cover the packages, their utility and how you could find new packages if you want to extend the existing system.

## 1. Finding the packages

As explain in the previous paragraph, the first thing to do to deploy the CESReS model is to find and load the required packages.
The clusters use modules.
They ensure compatibility between computation nodes and contain packages optimized for each cluster.
Further details are provided inside the CECI documentation (https://support.ceci-hpc.be/doc/_contents/UsingSoftwareAndLibraries/UsingPreInstalledSoftware/index.html). 
There you can also find the list of modules that could be loaded on the clusters.

We would like to mention that this module list has not been updated for a while, and does not match what is available on the clusters.
Using `module avail` on the `dragon2` interface will provide the effectively available modules.

The modules are also grouped into meta-modules representing their "release" date.
For example, the `Python/3.10.4-GCCcore-11.3.0` module is only available if the `releases/2022a` is loaded first.
Otherwise, the module loader will warn you that this module is not part of the currently considered releases, forcing you to consult the documentation with `module spider MODULE_YOU_WANT_TO_LOAD`.
During the experiments done for this thesis, careful choice of the modules was performed.
We ensured each to be compatible with each other, but also with the local deployment described in the `readme.md` file, and the packages imported by the programs.
If you want to use other releases or version of the modules, please consider looking at the dependencies required by the python packages.
Only modify these if you know what you are doing.
Otherwise, we cannot guarantee the program to continue working as it should.

Here is the list of the packages with a brief explanation of their respective usage.
- `Python/3.10.4-GCCcore-11.3.0`: version 3.10.4 of the Python interpreter
- `SciPy-bundle/2022.05-foss-2022a`: bundle of different packages used to perform scientific computation with Python (scipy, numpy, pandas)
- `CUDA/11.7.0`: interface with the GPUs to perform the training on these hardware pieces
- `OpenMPI/4.1.4-NVHPC-22.11-CUDA-11.7.0`: allows running threads and CUDA instructions (see https://www.open-mpi.org/ for more details)
- `tensorboardX/2.5.1-foss-2022a`: every package to use tensors with Python
- `PyTorch/1.12.1-foss-2022a-CUDA-11.7.0`: the 1.12.1 version of the Pytorch framework (loading CUDA and OpenMPI before this module is needed to ensure a working CUDA implementation)
- `TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0`: the 2.11.0 TensorFlow framework (needed for some additional packages)
- `torchvision/0.13.1-foss-2022a-CUDA-11.7.0`: the 0.13.1 TorchVision framework (needed for some additional packages)

Loading the complete set of modules can be performed by executing these lines:
```bash
module purge
module load releases/2022a
module load Python/3.10.4-GCCcore-11.3.0
module load SciPy-bundle/2022.05-foss-2022a
module load CUDA/11.7.0
module load OpenMPI/4.1.4-NVHPC-22.11-CUDA-11.7.0
module load tensorboardX/2.5.1-foss-2022a
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
module load torchvision/0.13.1-foss-2022a-CUDA-11.7.0
```

However, these loaded modules are not enough to execute the model program.
You will need a virtual environment containing additional packages, like AllenNLP or Captum.

NOTE: earlier, git was provided as a way to get the repository inside your personal environment in the CECI cluster.
However, git is not present by default. 
You could load `module load git/2.23.0-GCCcore-8.3.0-nodocs` to have access to it.

## 2. Creating a virtual environnement

To create the virtual environment, you will first need to load the previously mentioned modules.
Executing the provided lines should do the job.

On the CECI clusters, it is advised not to use conda.
A `virtualenv` is the encouraged alternative.

The following lines will create a virtual environnement named `cesres_env` and install the required packages.
The `CECI_requirements.txt` file will contain the information about the additional packages.

```shell
mkdir ~/cesres_env
virtualenv --system-site-packages ~/cesres_env/
source ~/cesres_env/bin/activate
pip install -r CECI_requirements.txt
deactivate
```

After these instructions, you will be able to perform `source ~/cesres_env/bin/activate` to activate the virtual environment 
containing all the required packages.

NOTE: If you were not able to install the whole set of packages by using the `pip install` command, split the requirements file 
into multiple subsets of dependencies.

## 3. Creating the configuration

Using the configuration wizard on the CECI website, you can create a job configuration structure to submit via slurm.
You should add the following components at the end of the provided instructions:
- load all the modules we previously listed
- activate the virtualenv
- create the output directories for the model
- use `cp` to copy the components (models, datasets, ...) from `$HOME/cesres` to the `$LOCALSCRATCH/$SLURM_JOB_ID`
- navigate to the copied files
- use `mpirun python3 cesres_graphcodebert_model.py ...` with our configuration parameters
- deactivate the virtual env (use `deactivate`)
- use `cp` to copy back all the results to the `$SLURM_SUBMIT_DIR` (the scratch is deleted avec the job execution)

## 4. Creating a slurm job

You can use `sbatch` to launch your script.

Enjoy !
