# ===========================================================
#
# Select random python code inside a directory
#
# Author: Guillaume Steveny
# Year: 2023 -- 2024
#
# ===========================================================

import signal
from argparse import ArgumentParser
import os
import shutil
from random import Random
import hashlib
from mutation_rule import NoMatches
from redbaron_rules import RemoveCommentsAndDocstrings, RenameAllDef, RenameAllVarsDummy


def try_create_dir(path: str) -> None:
    """
    Function to create a dict. Ignore the creation if it already exists

    Args:
        path: the path of the directory to create

    Returns:
        None
    """
    try:
        os.mkdir(path)
    except OSError as e:
        pass


# NOTE: this function could be improved
def modify_and_write(file: str, new_name: str) -> None:
    """
    Apply mutation rules to the content of file and write it in the output directory under the name new_name

    Args:
        file: the path to the original file to be mutated
        new_name: the new name the written file should have

    Returns:
        None
    """
    try:
        # Opens the original file in read mode, create the output file
        with open(file, "r", encoding="utf-8") as original_file, open(args.output+os.sep+new_name, "w") as output:
            # Add a signal to avoid mutation stopping
            signal.alarm(10)
            # Get the original code
            original_code = original_file.read()
            # Remove comments if user asked for it
            if args.no_comments:
                original_code = remove_docstrings.apply(original_code, ("", 0, ""))[2]
            # Modifies the variables and functions if user asked for it
            if args.dummy:
                original_code = dummy_variable.apply(original_code, ("", 0, ""))[2]
                original_code = dummy_function.apply(original_code, ("", 0, ""))[2]
            # Deactivates the alarm
            signal.alarm(0)
            # Ensure not having empty codes
            if len(original_code) == 0:
                return
            # Write the result in the output file
            output.write(original_code)
    # Avoids problem with the mutation rules
    except NoMatches:
        pass
    # Avoid problem with non-existing files
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    parser = ArgumentParser(description="Program allowing one to specify multiple directory and to randomly select "
                                        "a subset of files contained in this directories. A filtering on the "
                                        "extension of the files can be done to ensure having only files that are "
                                        "accepted by another program. "
                                        "Author: Guillaume Steveny ; Year: 2023-2024.")

    parser.add_argument("-e", "--extension", action="store", type=str, default=".py",
                        help="File extension the program should consider when sampling the complete content of "
                             "all the directories. (default=.py)")

    parser.add_argument("-f", "--filter", action="store", type=str, default=None,
                        help="String that should be contained inside the file paths. (default : no filtering)")

    parser.add_argument("-s", "--seed", action="store", nargs="?", type=int, default=None, const=42,
                        help="Seed for random sampling. If this value is unspecified, all the filtered files in the "
                             "directories will be copied in the output directory.")

    parser.add_argument("-p", "--partition", action="store", type=int, default=90,
                        help="Fraction of files to keep in total. Should be a number between 1 and 100. If not "
                             "the default value is used. This parameter is only activated when a seed is "
                             "specified. (default=90)")

    parser.add_argument("-l", "--limit", action="store", type=int, default=None,
                        help="Number of files maximum to keep. If this value is higher than the number of files found "
                             "all files are considered. (default : consider all files)")

    parser.add_argument("-ls", "--last_submission", action="store", nargs="?", type=str, default=None, const="success",
                        help="Parameter to specify if the last submission should only be considered when filtering the "
                             "sources. If this parameter is set, file names are expected to be formatted as: qx_N_"
                             "S_HASH.py where N is the student number, S its submission number for this question and "
                             "HASH its hash. Files to keep are inside the sub-folders of the same name than this "
                             "parameter if an additional value is set, otherwise, the default 'success' value is used."
                             "Being not compliant will lead to a crash of the tool.")

    parser.add_argument("-du", "--dummy", action="store_true",
                        help="Specifies if each file should be mutated with the RedBaron rules (cfr redbaron_rules.py) "
                             "rename_all_vars_dummy and rename_all_def. This would allow to wipe out the information "
                             "associated with variable names.")

    parser.add_argument("-nc", "--no_comments", action="store_true",
                        help="Specifies if the comments and docstring should be removed from each of the selected "
                             "files.")

    parser.add_argument("-a", "--anonym", action="store_true",
                        help="Indicates if the name of each file should be replaced by a MD5 hash of its original "
                             "name.")

    parser.add_argument("-r", "--rename", action="store_true",
                        help="Specify if the files should be renamed according to the sub-folder in which it was "
                             "originally found. The new names would be '_'.join(parent_dirs)+'_'name")

    parser.add_argument("-d", "--delete", action="store_true",
                        help="Specify if the output directory should be cleaned if it already exists.")

    parser.add_argument("output", action="store", type=str,
                        help="Output directory in which the files will be copied.")

    parser.add_argument("sources", action="store", nargs="*",
                        help="All the directories that should be considered for the sampling operation.")

    args = parser.parse_args()

    # Check if the directory should be cleaned
    if args.delete and os.path.isdir(args.output):
        shutil.rmtree(args.output)

    # Create the output directory
    try_create_dir(args.output)

    # Instantiate the random
    rd = None
    if args.seed is not None:
        rd = Random(args.seed)

    # Complete list of files
    all_files = []

    # For filtering only the last submission
    if args.last_submission:
        last_submission = {}

    # Docstring and comments removal
    remove_docstrings = RemoveCommentsAndDocstrings("", False, None) if args.no_comments else None
    # Dummy functions and variables
    dummy_function = RenameAllDef("", False, None) if args.dummy else None
    dummy_variable = RenameAllVarsDummy("", False, None) if args.dummy else None

    # Go over all the source
    for source in args.sources:
        # All the files contained in the source (allow recursive search)
        content = []
        for root, subdirs, files in os.walk(source):
            # Check if only last submission to keep
            if args.last_submission:
                for c in files:
                    if c.endswith(args.extension) and (args.filter is None or args.filter in root):
                        info = c[:-3].split("_")
                        if int(info[2]) > last_submission.get(info[3], (-1,))[0]:
                            if "success" in root:
                                last_submission[info[3]] = (int(info[2]), "success", root+os.sep+c)
                            else:
                                last_submission[info[3]] = (int(info[2]), "failed", root+os.sep+c)

            filter_files = [root+os.sep+c for c in files if c.endswith(args.extension) and
                            (args.filter is None or args.filter in root)]
            content.extend(filter_files)
        # Accumulate all the files
        all_files.extend(content)

    # If the last submission should be kept, update all_files
    if args.last_submission:
        all_files = [info[2] for key, info in last_submission.items() if info[1] == args.last_submission]

    # Information to the user
    print(f"Found {len(all_files)} files in the source directories.")

    # Get the partition
    partition = args.partition
    if partition < 1 or partition > 100 or args.seed is None:
        partition = 100

    # Sample the files
    if rd is not None and partition != 100:
        part = (len(all_files) * partition) // 100
        all_files = rd.sample(all_files, part)
        print(f"Sampling done, kept {len(all_files)} files.")

    # Get the limit
    limit = args.limit if args.limit is not None else len(all_files)
    if rd is not None and limit < len(all_files):
        rd.shuffle(all_files)
    all_files = all_files[:limit]
    print(f"Kept {len(all_files)} files out of the sampling.")

    # Copy each file into the output
    for file in all_files:
        if args.rename or args.anonym:
            dir_name_file = file.split(os.sep)
            original_dir = os.sep.join(dir_name_file[:-1])
            # If the files should be anonym, replace the name by the MD5 hash of the name
            if args.anonym:
                new_name = hashlib.md5(dir_name_file[-1].encode("utf-8")).hexdigest()+args.extension
                new_name = '_'.join(dir_name_file[:-1]) + '_' + new_name
            # Otherwise, the files have a constructed name
            else:
                new_name = '_'.join(dir_name_file)
            # Check if modification should be done on the input code
            # If not, the file is only copied
            if not args.no_comments and not args.dummy:
                shutil.copy(file, original_dir+os.sep+new_name)
                shutil.move(original_dir+os.sep+new_name, args.output)
            # Otherwise, it is already mutated accordingly
            else:
                modify_and_write(file, new_name)
        else:
            # Check if modification should be done on the input code
            # If not, the file is only copied
            if not args.no_comments and not args.dummy:
                shutil.copy(file, args.output)
            # Otherwise, it is already mutated accordingly
            else:
                modify_and_write(file, file.split(os.sep)[-1])

    print("All files copied. Program terminated.")
