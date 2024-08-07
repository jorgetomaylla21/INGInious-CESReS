# ===========================================================
#
# Mutation merger and resampler
#
# Author: Guillaume Steveny
# Year: 2023 -- 2024
#
# ===========================================================

import warnings
import numpy.random as rd
from sklearn.model_selection import train_test_split
import os
import argparse


def try_create_dir(path: str) -> None:
    """
    Function to create a dict. Ignore the creation if it already exists.

    Args:
        path: the path of the directory to create.

    Returns:
        None
    """
    try:
        os.mkdir(path)
    except OSError as e:
        pass


def write_to_file(output: str, codes: list[tuple[str, int, str, str]]) -> None:
    """
    Write a file in a CESReS format with the codes in the codes list.

    Args:
        output: the name of the path where the output files should be written to
                (supposed a directory already existing).
        codes: the output (or a part) of the process_all_files function.

    Returns:
        None
    """
    with open(output, "w") as file:
        # Write each transformed codes according to the format used by CESReS
        for code in codes:
            file.write(code[2] + "\n $x$ " + code[3] + "\n$$$\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Program allowing to resample and merge multiple all_codes.txt files "
                                                 "obtained by using the mutation_labelling program. The two modes can "
                                                 "be used separately by using the correct parameters."
                                                 "Author: Guillaume Steveny ; Year: 2023-2024")

    parser.add_argument("-m", "--merge", action="store_true",
                        help="Specify if the multiple file paths given as input should be merged or not. If this "
                             "parameter is not set, the number of file paths should be of one single directory path.")

    parser.add_argument("-r", "--resample", action="store_true",
                        help="Specify if the source codes should be resampled into a train, a validation and a test "
                             "set. Otherwise, a single all_codes.txt file is created in the output path.")

    parser.add_argument("-l", "--limit", action="store", type=int, default=None,
                        help="Limit on the number of different original file (file from which the mutants are "
                             "generated) to keep. (default : keep_all)")

    parser.add_argument("-k", "--k_subset", action="store", type=int, default=None,
                        help="Allows to specify the maximal number of code to sample for each name. "
                             "(default : keep all)")

    parser.add_argument("-w", "--weight_sample", action="store_true",
                        help="This parameter is ignored unless used with the k argument. It allows to gave more "
                             "weights to the labels that are unrepresented in the dataset to aim a coherent and "
                             "uniform label distribution")

    parser.add_argument("-nd", "--no_duplicate", action="store_true",
                        help="Apply a procedure to avoid having multiple times the same labels for the mutants. "
                             "This will only keep one example for each label set for each original code.")

    parser.add_argument("-t", "--train_split", action="store", type=int, default=90,
                        help="Ensure creating a train, a validation and a test file containing subset of the generated "
                             "mutants. The value to be passed to this argument is the percentage (expressed between 0 "
                             "and 100) to be use when splitting the train set and validation set from the test set. "
                             "(default = 90)")

    parser.add_argument("-o", "--output", action="store", type=str, default="output/modified_code",
                        help="The directory name in which the resulting 'txt' files should be stored after the "
                             "merge and resample. It is not mandatory for you to create it in advance, the program "
                             "will try to do it for you without suppressing the contained data if it already exists. "
                             "If you only chose the merge, a single 'all_codes.txt' file will be created there, "
                             "otherwise, the splits are also created. " 
                             "(default = output/modified_code)")

    parser.add_argument("-s", "--seed", action="store", type=int, default=33,
                        help="The random seed that should be used if the resampling procedure was selected. "
                             "(default = 33)")

    parser.add_argument("source", action="store", type=str, nargs='+',
                        help="Path of the different directories in which the 'all_codes.txt' files can be found. It "
                             "should be noted that these should only be the directory paths not the path of the txt "
                             "files.")

    args = parser.parse_args()

    # The file paths
    paths = args.source
    filename = "all_codes.txt"
    output_path = args.output

    # Should we merge ?
    merging = args.merge
    if not merging and len(paths) > 1:
        raise ValueError("If you decide to not merge, you should not give multiple directories to this program")

    # Try to create the output directory
    try_create_dir(output_path)

    # To store the codes for each original file
    codes = {}
    list_codes = []

    # Count the labels (if used to have subset weighted)
    labels = {}
    total_labels = 0

    # Should remove duplicated labels
    no_duplicate = args.no_duplicate

    # For each directory referenced
    for path in paths:
        # Get the file content
        with open(path + "/" + filename, "r") as file:
            text = file.read()

        # Get the mutants (separated by $$$)
        mutants = text.split("\n$$$\n")[:-1]

        # Go over all the generated mutants
        for mut in mutants:
            code, info = mut.split(" $x$ ")
            label, name, number = info.split(" $ ")

            if name not in codes:
                codes[name] = {} if no_duplicate else []

            if no_duplicate and label in codes[name]:
                continue
            elif no_duplicate:
                codes[name][label] = (name, number, code, label)
            else:
                codes[name].append((name, number, code, label))

            labels[label] = labels.get(label, 0) + 1
            total_labels += 1

        # If a limit is applied on the name to consider
        if args.limit is not None:
            seed = args.seed
            # Random element to sample from the codes sets
            rd.seed(seed)
            # Select a subset of the names
            names = rd.choice(list(codes.keys()), args.limit, replace=False)
        else:
            names = list(codes.keys())

        # Get all the codes subsets
        list_codes = [list(codes[name].values()) if no_duplicate else codes[name] for name in names]

    if merging:
        print("Merging the source files")
        # Save all the codes in a different manner
        all_codes = []
        # Create new labelling for the codes which is LABEL $ FILE_NAME $ MUTANT_NUMBER
        for cs in list_codes:
            current_codes = []
            for i, c in enumerate(cs):
                current_codes.append((c[0], c[1], c[2], f"{c[3]} $ {c[0]} $ {i}"))
            all_codes.extend(current_codes)
        # Write the all_codes.txt file to be possibly resampled afterwards
        write_to_file(output_path + "/" + "all_codes.txt", all_codes)

    # Check the resampling parameter
    resampling = args.resample
    if resampling:
        print("Resampling the source files")

        # Set the seed and size of splits
        seed = args.seed
        size = args.train_split / 100

        # Check if the files should be sampled
        if args.k_subset:
            # Random element to sample from the codes sets
            rd.seed(seed)
            # Temporary list that will replace list_codes after
            tmp_list = []

            # Go over the codes, and sample them
            for c in list_codes:
                # If weights should be added, each code is considered with a higher weight if it is less frequent
                if args.weight_sample and len(c) >= args.k_subset:
                    tot_labels_example = sum([labels[e[3]]/total_labels for e in c])
                    probs = [(1 - labels[e[3]]/total_labels)/(len(c) - tot_labels_example) if len(c) > 1 else 1
                             for e in c]
                    sampled_index = rd.choice([i for i in range(len(c))], args.k_subset, replace=False,
                                              p=probs)
                # Otherwise, the distribution is uniform
                else:
                    if len(c) < args.k_subset:
                        sampled_index = [i for i in range(len(c))]
                    else:
                        sampled_index = rd.choice([i for i in range(len(c))], args.k_subset, replace=False)
                # Get the list of codes according to the selected indexes
                sampled_list = [c[i] for i in sampled_index]
                tmp_list.append(sampled_list)
            list_codes = tmp_list

        # If the size is not correct
        if size <= 0 or size > 1.0:
            warnings.warn("The split size you chose is not a correct value (between 1 and 99), the default value "
                          "will be used instead (90).")
            size = 0.9

        # Split the values
        train_codes, test_codes = train_test_split(list_codes, train_size=size, random_state=seed)
        train_codes, validation_codes = train_test_split(train_codes, train_size=size, random_state=seed)

        # Write each type of files
        for output_name, output_list in zip(["train", "validation", "test"], [train_codes, validation_codes, test_codes]):
            output = output_path + "/" + output_name + ".txt"
            write_to_file(output, [code for l in output_list for code in l])


