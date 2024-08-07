# Mutation Labelling Program

Author: Guillaume Steveny

Year: 2023 - 2024

## 1. Installing the packages

To run the programs, you need to have Python 3.8 or 3.9 on a Linux distribution.

You should then follow the instructions of Comby to install their tool: https://comby.dev/docs/get-started

Finally, you should have the following packages installed:
- comby (https://pypi.org/project/comby/)
- numpy
- scikit-learn
- tqdm
- redbaron (https://pypi.org/project/redbaron/)

Installing the `cesres` environment we described in the "installation" directory of this repository will 
provide the required packages without further manipulations.


## 2. Running a demo

You will find an exmaple ruleset in the file "ruleset_example.txt".

You can use the following command to modify the files contained in test_files with this ruleset and write the results in a newly created output directory.
> python3 mutation_labelling.py -r -o output -s test_files ruleset_example.txt

Using `-h` will show the small documentation of the main program.

The file "mutation_grammar.txt" gives the additional information about creating the rule sets.


## 3. Utility functions

The "utils" directory gives you access to additional programs we created to check and operate on mutants.
- `check_mutants.py` takes a directory and gives the label distribution of the generated datasets
- `resample_mutants.py` can merge multiple generated datasets and select instances randomly from these
- `select_random_directory.py` randomly selects files inside a directory with possible pre-processing steps before mutation

The `test_redbaron_rules.py` file contains the unit tests we developed to assert the validity of our rules using RedBaron.