# ===========================================================
#
# Mutation labelling program - Main program
#
# Author: Guillaume Steveny
# Year: 2023 -- 2024
#
# ===========================================================

from __future__ import annotations

import os
import hashlib
import subprocess
from multiprocessing.pool import ThreadPool, ApplyResult
from multiprocessing.context import TimeoutError

import attr
from comby import Comby, LocationRange, Match
import argparse
from typing import Callable, Optional, Iterator, Type
import numpy as np
from comby.exceptions import CombyBinaryError
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm

from mutation_rule import AbstractRule, NoMatches
from redbaron_rules import possible_redbaron_rules, RedBaronRule

rd = random.Random()
nr = np.random

# -------------------------------------------------------------------


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

# -------------------------------------------------------------------


# We override the Comby object by creating a subclass of it.
# This strategy allows to add the attribute timeout for stopping the process when it does not answer.
@attr.s(frozen=True, slots=True)
class CombyTimeout(Comby):
    """
    Special modification by inheritance of the Comby object to add a timeout inside the call method
    This new call method will be used by the matches and rewrite methods of the original Comby object
    """
    timeout = attr.ib(type=int, default=10)

    # The method copies the original code from Comby.
    # We modified the call to subprocess run to add the timeout.
    # We also added the NoMatches thrown when the timeout expires.
    def call(self, args: str, text: Optional[str] = None) -> str:
        """Calls the Comby binary.

        Arguments
        ---------
        args: str
            the arguments that should be supplied to the binary.
        text: Optional[str]
            the optional input text that should be supplied to the binary.

        Returns
        -------
        str
            the output of the execution.

        Raises
        ------
        CombyBinaryError
            if the binary produces a non-zero return code.
        NoMatches
            if the timeout has expired when calling the binary
        """
        input_ = None
        if text:
            input_ = text.encode('utf8')

        # args = "-match-newline-at-toplevel " + args
        cmd_s = f'{self.location} {args}'
        try:
            p = subprocess.run(cmd_s,
                               shell=True,
                               stderr=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               input=input_,
                               timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise NoMatches()

        err = p.stderr.decode('utf8')
        out = p.stdout.decode('utf8')

        if p.returncode != 0:
            raise CombyBinaryError(p.returncode, err)
        return out

# -------------------------------------------------------------------


def check_matches(source: str, pattern: str, comby: CombyTimeout):
    """
    Function to check if there is at least one match of a specific pattern

    Args:
        source:  a str representing the code to be modified
        pattern: a str representing the Comby pattern to be matched in the source
        comby:   a CombyTimeout object to operate the mutation

    Returns:
        True if there is at least one match

    Raises:
        NoMatches if there is indeed no match for this pattern
        This exception is also raised if the timeout is expired by the Comby process
    """
    matches = [m for m in comby.matches(source, pattern, language=".py")]
    if len(matches) > 0:
        return True
    raise NoMatches()


def apply_once(source: str, pattern: str, replace: str, comby: CombyTimeout) -> str:
    """
    Function to apply a mutation to a random match of a pattern in a source code

    Args:
        source:  a str representing the code to be modified
        pattern: a str representing the Comby pattern to be matched in the source
        replace: a str containing the Comby format used to rewrite a match of the pattern
        comby:   a CombyTimeout object to operate the mutation

    Returns:
        the mutated code where either the first match is replaced (if not random) or a random one

    Raises:
        NoMatches if there is indeed no match for the pattern
        This exception is also raised if the timeout is expired by the Comby process
    """

    # Get all the matches and select a random one
    matches = list(comby.matches(source, pattern, language=".py"))

    # Check if there is at least one match
    if len(matches) == 0:
        raise NoMatches()

    # Select a match to modify
    m: Match = rd.choice(matches) if args.random is not None else matches[0]

    # Get the location of the match and modifies this part
    loc: LocationRange = m.location
    part = source[loc.start.offset:loc.stop.offset]
    new_part = comby.rewrite(part, pattern, replace, language=".py")

    # Return the newly constructed code
    return source[:loc.start.offset] + new_part + source[loc.stop.offset:]


def apply_all(source: str, pattern: str, replace: str, comby: CombyTimeout) -> str:
    """
    Function to transform a source code according to the replacement pattern for all the matches found

    Args:
        source:  a str representing the code to be modified
        pattern: a str representing the Comby pattern to be matched in the source
        replace: a str containing the Comby format used to rewrite a random match of the pattern
        comby:   a CombyTimeout object to operate the mutation

    Returns:
        The mutated code where each match has been replaced

    Raises:
        NoMatches if there is indeed no match for the pattern
        This exception is also raised if the timeout is expired by the Comby process
    """
    # Check if there is at least one match
    check_matches(source, pattern, comby)

    return comby.rewrite(source, pattern, replace, language=".py")


# Different rules that could be applied (<> is not usable by the user)
rule_appliers = {
    " <> ": lambda src, pat, rep, c: src,
    " -> ": apply_once,
    " => ": apply_all
}

# -------------------------------------------------------------------


class Rule(AbstractRule):
    """
    Class representing a rule to be applied by the mutator

    Attributes:
        lhs:     a str of the Comby pattern to be match and replaced by the rule
        applier: the rule applier created by using -> or => in the rule list file
        rhs:     a string of the Comby pattern which will replace lhs
        label:   a string containing the label to be associated with the mutant after the modification
        mode:    an int representing the mode of the rule (0 indicates only one possible replace, 1 indicates multiple
                 possible replaces)
        comby:   a CombyTimeout object to operate the mutation
        expr:    the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        lhs:     a str of the Comby pattern to be match and replaced by the rule
        applier: the rule applier created by using -> or => in the rule list file
        rhs:     a str of the Comby pattern which will replace lhs
        label:   a str containing the label to be associated with the mutant after the modification
        comby:   a CombyTimeout object to operate the mutation
        expr:    the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    __slots__ = ["lhs", "applier", "rhs", "label", "mode", "comby", "expr"]

    def __init__(self, lhs: str, applier: Callable[[str, str, str, CombyTimeout], str], rhs: str, label: str,
                 comby: CombyTimeout, expr: str = ""):

        # Create the label and expr attributes
        super().__init__(label, expr)

        # Map the attributes to their value
        self.lhs = lhs
        self.applier = applier
        self.rhs: str | list[str] = rhs
        self.mode = 0  # By default, this is a 0 mode rule
        self.comby = comby

        # Check the presence of || as symbols in the right-hand side to indicate a 1 mode rule
        if " || " in self.rhs:
            # The right-hand side becomes the list of possible replacement
            self.mode = 1
            self.rhs = self.rhs.split(" || ")

        # The $c{A|B|...} is an alternative to the || for a part of the expression without rewriting the rest
        elif "$c" in self.rhs:
            # Check there is indeed a single $c{} in the right-hand side (multiple not implemented for now)
            assert self.rhs.count("$c") == 1, "Only one $c per right-hand side for now"

            # Split the LEFT $c{A|...} END into LEFT A|... and END
            left, right = self.rhs.split("$c{")
            choice, end = right.split("}")

            # Get the different possible values specified in the $c
            vals = choice.split("|")

            # The possible rhs are LEFT A END for all the A in the $c
            self.rhs = [left + v + end for v in vals]
            self.mode = 1

        # Otherwise, the rule is a classical 0-mode rule
        else:
            # Ensure the right values to be stored as attributes
            self.mode = 0
            self.rhs = self.rhs
            # Here we check the presence of a blank character, which will be replaced after using comby
            if "$b" in self.rhs:
                self.mode = 3

    def apply(self, src: str, info: tuple[str, int, str]) -> tuple[str, int, str, str]:
        """
        Method to apply the rule specified by this object and get the result and corresponding label

        Args:
            src:   a str representing the code to be modified
            info:  a tuple containing the name of the original file which contained the code to be modified,
                   the index of the applied rule and the previous labels associated with this code

        Returns:
            a tuple containing
                <ul>
                    <li> the name of the file (info[0])
                    <li> the rule number (only have sense when doing not random choices) (info[1])
                    <li> the mutated code
                    <li> the label associated (if there is multiple labels, are separated by " ; ")
                </ul>

        Raises:
            NoMatches if the applier throws such error which should be handled when applying the rules
        """
        # Create the new label
        label = self.construct_label(info)

        # If the mode is the classical one, apply simply
        if self.mode == 0:
            return info[0], info[1], self.applier(src, self.lhs, self.rhs, self.comby), label

        # Otherwise, a possible replace should be selected randomly if specified by the user (if not take the first
        # choice)
        elif self.mode == 1:
            rhs = rd.choice(self.rhs) if args.random is not None else self.rhs[0]
            return info[0], info[1], self.applier(src, self.lhs, rhs, self.comby), label

        # We get the result of the applier and replace all the blank ($b) characters
        else:
            r = self.applier(src, self.lhs, self.rhs, self.comby)
            r = r.replace("$b", "")
            return info[0], info[1], r, label

    def __str__(self):
        # Check if the line creating this rule was specified, if not, print a kind of representation of what it was
        if self.expr == "":
            return f"Rule: {self.lhs} (-|=)> {self.rhs} ; {self.label}"
        else:
            return self.expr

    def __repr__(self):
        return str(self)


# -------------------------------------------------------------------


def parse_rules(path: str, comby: CombyTimeout) -> dict[str, tuple[int, list[AbstractRule]]]:
    """
    Gets the interpretable list of rules to be used to transform the codes

    Args:
        path:  a str representing the name of the file containing the specified rules
        comby: the CombyTimeout object used to mutate the code (used only for the Comby rules)

    Returns:
        a dict where the key are the labels and the items are tuples containing:
            <ul>
                <li> the number of rules that can be selected
                <li> the list of the AbstractRule objects to be applied for this label
            </ul>
    """
    # Get the lines of the rules
    with open(path, "r") as file:
        lines = file.readlines()
    # Ignore empty lines
    lines = [line.strip() for line in lines if len(line.strip()) > 0]

    # Rule indicator
    rules_specifier = {"$", "£"}

    # The rule not specified is the identity function
    rules: dict[str, tuple[int, list[AbstractRule]]] = {}

    # Store if we are currently in a set rule or not
    toggle_rule_set = False

    # The current label we are dealing with
    label = ""

    # For each rule, try to parse it
    for line in lines:
        # If the line start with a comment symbol, we ignore it
        if line[0] == "#":
            continue

        # If the line specifies a rule but without specifying the associated label before, ignore it
        if line[0] in rules_specifier and not toggle_rule_set:
            continue

        # If the line indicates we ended the rules for this label, go to the next line
        if ";" == line[0]:
            toggle_rule_set = False
            continue

        # If the line is not a rule specifier, and we are not in a rule set already (otherwise, the line is ignored)
        if line[0] not in rules_specifier and not toggle_rule_set:
            # Get the label and the count linked with this new rule set
            label, count = line.split()

            # Update the rules output
            rules[label] = (int(count), rules.get(label, (0, []))[1])

            # Activate the rule set
            toggle_rule_set = True

        # Otherwise, if the line starts with $ in a rule set, parse this rule
        elif line[0] == "$" and toggle_rule_set:
            # We get rid of the 4 first chars ($ and 3 spaces)
            expr = line[4:]

            # Choose the correct rule applier
            if "=>" in line:
                lhs, rhs = expr.split(" => ")
                rule = rule_appliers[" => "]
            elif "->" in line:
                lhs, rhs = expr.split(" -> ")
                rule = rule_appliers[" -> "]
            else:
                # If the rule is badly formatted, we ignore it
                continue

            # Transform escaped characters to ensure not breaking the Comby expressions
            for s, t in [("\\n", "\n"), ("\\-", "\-"), ("\\=", "\="), ("\\>", "\>")]:
                lhs = lhs.replace(s, t)
                rhs = rhs.replace(s, t)

            # Create a rule object to end the parsing of the rule parts
            try:
                rules[label][1].append(Rule(lhs, rule, rhs, label, comby, expr))
                print(f"Parsed {expr} for label {label}")
            except Exception as e:
                # If the rule parsing failed, ignore the line but show the reason to the user
                print(e)
                continue

        elif line[0] == "£" and toggle_rule_set:
            # We get rid of the 4 first chars ($ and 3 spaces)
            expr = line[4:].strip()

            # Use the rule set and the mappings
            try:
                rule_object: Type[RedBaronRule] = possible_redbaron_rules[expr.split(" ")[0]]
                rules[label][1].append(rule_object(label, args.random is not None, rd, expr))
                print(f"Parsed {expr} for label {label}")
            except Exception as e:
                # If the rule has a problem when created, indicate to the user it should only pick from the list
                print("The rule you decided to specify does not exist. Please consider using one of the following: "
                      "[" + ", ".join(list(possible_redbaron_rules.keys())) + "]. You gave "+expr)
                continue

    return rules

# -------------------------------------------------------------------


def tgz_iterator(path: str) -> Callable[[], Iterator[tuple[str, str, str]]]:
    """
    Function creating an iterator to go over the files indicated inside a tgz file
    The tgz archive need to be retrieved from Inginious to be correctly formatted
    The considered compressed files are yaml ones where the student code can be found in the input/q1 subkey

    Args:
        path: the string representing the name of the tgz archive to be explored (if the file is not a .tgz one,
              the default file ../output/submissions.tgz will be used)

    Returns:
        A tuple iterator were each output is a tuple composed of three strings. One for the name of the compressed file,
        one for the code linked with this file and one for the previously associated label (here = '')
    """
    def ite() -> Iterator[tuple[str, str, str]]:
        """
        Iterator for outputting the codes found inside an Inginious tgz archive

        Yields:
            a tuple composed of three string entries: the first is the name of the yaml file in which the code
            referenced by the second entry was found and the last is an empty string.
        """
        # Import the needed module for this task
        import tarfile
        import yaml

        # Open the tar file
        t = tarfile.open("../output/submissions.tgz" if path.endswith(".tgz") else path)

        # Progress bar
        file_limit = args.number_files
        tq = tqdm(total=file_limit if file_limit > 0 else 0)
        i = 0

        # Specify the filter that should be checked on compressed files
        name_filter = args.filter if args.filter is not None else "/submission.test"

        # We go over the members of the tar file
        try:
            for m in t:
                try:
                    # If the member is an Inginious submission file
                    if name_filter in m.name:
                        # Extract the file
                        e = t.extractfile(m)
                        # Decode it in utf-8 and ensure unix like line end
                        f = e.read().decode("utf-8").replace("\r\n", "\n")
                        # The member should be closed
                        e.close()
                        # Get the yaml formatted file
                        d = yaml.safe_load(f)
                        # Get the name of the member
                        name = m.name
                        # Transforms the name if the anonym arg is set
                        if args.anonym:
                            name = hashlib.md5(name.encode("utf-8")).hexdigest()
                        # Yield the name of this file and its code
                        yield name, d["input"]['q1'], ''

                # Terminating with ctrl-c will enter here
                except KeyboardInterrupt as k:
                    tq.close()
                    t.close()
                    return
                # This should be present in an iterator to close it safely
                except GeneratorExit as g:
                    tq.close()
                    t.close()
                    return
                # Other exception are ignored
                except Exception as e:
                    pass

                # Update the progress bar and check the limit
                tq.update()
                i += 1
                if i == file_limit:
                    break

        # Terminating with ctrl-c will enter here
        except KeyboardInterrupt as k:
            tq.close()
            t.close()
            return
        # This should be present in an iterator to close it safely
        except GeneratorExit as g:
            tq.close()
            t.close()
            return
        # Other exception are ignored
        except Exception as e:
            pass

        # All the files have been processed
        tq.close()
        t.close()

    # Return the previously defined inner function
    return ite


def path_iterator(path: str) -> Callable[[], Iterator[tuple[str, str, str]]]:
    """
    Function allowing to recover an iterator on the python files stored in a directory

    Args:
        path: the string representing the name of the directory to be explored

    Returns:
        A tuple iterator were each output is a tuple composed of three strings. One for the name of the source file,
         one for the code found in this file and one for a previously associated label (here = '')
    """
    def ite() -> Iterator[tuple[str, str, str]]:
        """
        Iterator for outputting the codes found inside a directory

        Yields:
            a tuple composed of three string entries: the first is the name of the file in which the code referenced
            by the second entry was found, the last is an empty string
        """
        # Limit on number of files
        file_limit = args.number_files
        file_list = os.listdir(path)
        if file_limit >= 0:
            # Shuffle the list to ensure randomness
            if args.random is not None:
                rd.shuffle(file_list)
            file_list = file_list[:file_limit]

        # Extension filter on the files
        file_filter = args.filter if args.filter is not None else '.py'

        # Go over the (possibly) restricted file list
        try:
            for f in tqdm(file_list):
                try:
                    if f.endswith(file_filter):
                        # Get the code for this file
                        with open(path + "/" + f, "r") as file:
                            code = file.read()
                        # Transforms the name if the anonym arg is set
                        if args.anonym:
                            f = hashlib.md5(f.encode("utf-8")).hexdigest()
                        yield f, code, ''
                # Ensure a neat end of the iterator
                except GeneratorExit as g:
                    return
                except KeyboardInterrupt as k:
                    return
                except Exception as e:
                    pass
        except GeneratorExit as g:
            return
        except KeyboardInterrupt as k:
            return
        except Exception as e:
            pass

    # Return the previously defined iterator
    return ite


def mutant_iterator(path: str) -> Callable[[], Iterator[tuple[str, str, str]]]:
    """
    Function allowing to construct an iterator over an "all_codes.txt" file generated by this program. This would
    allow generating multiple mutation over the same codes.

    Args:
        path: the string representing the path (complete) of the "all_codes.txt" file that should be used

    Returns:
        A tuple iterator were each output is a tuple composed of three strings. One for the name of the source file,
         one for the code found in this file and one for a previously associated labels
    """
    def ite() -> Iterator[tuple[str, str, str]]:
        """
        Iterator for outputting the codes found inside an "all_codes.txt" file

        Yields:
            a tuple composed of three string entries: the first is the name of the file in which the code referenced
            by the second entry was found, the last is the labels previously associated with this example
        """
        # Get the complete content of the file
        with open(path, "r") as file:
            content = file.read()

        # Split all the examples by using "\n$$$\n" as separator (supposed to be in a CESReS format)
        examples = content.split("\n$$$\n")[:-1]

        # Limit on number of files
        file_limit = args.number_files
        if file_limit >= 0:
            # Shuffle the list to ensure randomness
            if args.random is not None:
                rd.shuffle(examples)
            examples = examples[:file_limit]

        # Go over the entries and output them one at a time
        for entry in tqdm(examples):
            try:
                # Get the information about this entry
                code, info = entry.split(" $x$ ")

                # Split the information
                labels, name, number = info.split(" $ ")

                # Transforms the name if the anonym arg is set
                if args.anonym:
                    name = hashlib.md5(name.encode("utf-8")).hexdigest()

                # Yield the complete entry to be processed
                yield name, code, labels

            except KeyboardInterrupt as k:
                return

            # Ensure a neat end of the iterator
            except GeneratorExit as g:
                return

    # Return the previously defined iterator
    return ite


def process_a_file(
        labels: list[str],
        rules: dict[str, tuple[int, list[AbstractRule]]],
        code: str,
        f: str,
        prev_label: str
) -> list[tuple[str, int, str, str]]:
    """
    Apply rules on a specific code which was found in a file

    Args:
        labels: the labels string used as key in the rules' dictionary
        rules:  the rules constructed by the parse_rules function
        code:   a str containing the code to be mutated
        f:      a str representing the name of the file to which the code belong
        prev_label: a str corresponding to a possible previous label associated with this code
                    (allows multiple mutation)

    Returns:
        a list where each element is a tuple containing
            <ul>
                <li> the name of the file
                <li> the rule number
                <li> the mutated code
                <li> the label associated (if there is multiple labels, are separated by " ; ")
            </ul>
    """
    # Result obtained when processing this file
    results = []

    # Shuffle the rules if they should be considered randomly
    applied_rules = 0

    # The labels are randomized only if the parameter random was specified
    random_labels = rd.sample(labels, len(labels)) if args.random is not None else labels

    # Go over the labels randomly
    for label in random_labels:
        # Try to apply the rule (hoping at least one match)
        # We randomized the rules is the parameter specified it
        random_rules = rd.sample(rules[label][1], len(rules[label][1])) if args.random is not None else rules[label][1]

        # Count the number of rule of this type
        label_count = 0
        max_label = rules[label][0]

        # Go over a random sampling of the rules
        for rule in random_rules:
            try:
                new_code = rule.apply(code, (f, applied_rules, prev_label))
                if len(new_code) > 0:
                    results.append(new_code)
                    applied_rules += 1
                    label_count += 1
            except CombyBinaryError as c:
                # We ignore errors in the return of comby
                pass
            except NoMatches as e:
                # When there is no match, we ignore this rule for this file
                # This ensures not having unmodified code labelled as modified
                pass
            except KeyboardInterrupt as k:
                return results

            # Check if rules of this label could be applied again
            if label_count == max_label or applied_rules == args.limit:
                break

        # If the number of applied rules is sufficient, go to the next file
        # We limit if we went above the limit for each code
        if args.limit == applied_rules:
            break

    # A shuffle is done to prevent the labels to be selected (only if random)
    if args.random is not None:
        rd.shuffle(results)
    return results


def process_all_files(
        path: str | None,
        rules: dict[str, tuple[int, list[AbstractRule]]]
) -> list[list[tuple[str, int, str, str]]]:
    """
    Function to apply the specified rules to each file in a directory
    The rules are selected randomly if random argument is set
    The mutated code are limited by the limit argument for each file

    Args:
        path:  a str representing the path to the directory to be used as reference
        rules: the rules constructed by the parse_rules function

    Returns:
        a list of lists (one for mutated original code) where each element is a tuple containing
            <ul>
                <li> the name of the file
                <li> the rule information
                <li> the mutated code
                <li> the label associated (if there is multiple labels, are separated by " ; ")
            </ul>
    """
    # List of all the resulting codes
    results = []
    tmp: list[ApplyResult[list[tuple[str, int, str, str]]]] = []

    # Create a threading pool to process the files asynchronously
    th_pool = ThreadPool(args.processes)

    # Get the iterator for having the files
    if path.endswith(".tgz") or args.extract:
        ite = tgz_iterator(path)
    elif path.endswith(".txt"):
        ite = mutant_iterator(path)
    else:
        ite = path_iterator(path)

    # Get the possible labels
    labels = list(rules)

    # Go over the files created by the correct iterator and process them one by one
    for f, code, prev_label in ite():
        try:
            tmp.append(th_pool.apply_async(process_a_file, (labels, rules, code, f, prev_label)))
        except KeyboardInterrupt as k:
            # If the user tries to stop the creation of the threads
            # We switch to the recovery of the results for each thread
            # This can be also stopped by a new ctrl-c
            print("Stopping the thread creation. Recovery of their results."
                  "A new ctrl+c allows to stop this and to save directly the processed codes.")
            break

    # Go through the thread to recover their results
    p: ApplyResult[list[tuple[str, int, str, str]]]
    for p in tqdm(tmp):
        try:
            # Timeout on the get to be sure to avoid infinite runs
            r = p.get(len(rules)*args.max_time)
            results.append(r)
        except KeyboardInterrupt as k:
            # If the user want to skip the result recovery, switch to the file saving
            return results
        except TimeoutError:
            # If a timeout was encountered, go just to the next thread, ignoring this unsuccessful one
            pass

    # Terminate the pool
    th_pool.terminate()

    return results

# -------------------------------------------------------------------


def write_to_file(output: str, codes: list[tuple[str, int, str, str]]) -> None:
    """
    Write a file in a CESReS format with the codes in the codes list

    Args:
        output: the name of the path where the output files should be written to
                (supposed a directory already existing)
        codes: the output (or a part) of the process_all_files function

    Returns:
        None
    """
    with open(output, "w") as file:
        # Write each transformed codes according to the format used by CESReS
        for code in codes:
            if len(code[2]) > 0:
                file.write(code[2] + "\n $x$ " + code[3] + "\n$$$\n")


def to_cesres_file(
        transformed_codes: list[list[tuple[str, int, str, str]]],
        output_path: str
) -> None:
    """
    Function to format the different mutated codes in a format accepted by the CESReS model

    Args:
        transformed_codes: the output of the process_all_files function
        output_path: the name of the path where the output files should be written to
                     (supposed a directory already existing)

    Returns:
        None
    """
    # Indicate we switch to the file writing
    print("Saving codes")

    # Save all the codes in a different manner
    all_codes = []
    # Create new labelling for the codes which is LABEL $ FILE_NAME $ MUTANT_NUMBER
    for cs in transformed_codes:
        current_codes = []
        for i, c in enumerate(cs):
            current_codes.append((c[0], c[1], c[2], f"{c[3]} $ {c[0]} $ {i}"))
        all_codes.extend(current_codes)
    # Write the all_codes.txt file to be possibly resampled afterwards
    write_to_file(output_path + "/" + "all_codes.txt", all_codes)

    # Check if there should be a random splitting
    if args.train_split is not None:
        # Get the size and check its value
        size = args.train_split / 100
        if size <= 0 or size > 1.0:
            # Error in split, but save the mutants however
            output = output_path + "/" + "results.txt"
            transformed_codes = [code for list_codes in transformed_codes for code in list_codes]
            write_to_file(output, transformed_codes)
            raise ValueError(f"The split should be a valid percentage, which is not the case here. "
                             f"Received {size*100}."
                             f"The code were saved anyway in {output}.")

        # Set a random seed if not specified
        if args.random is None:
            seed = 42
        else:
            seed = args.random

        try:
            # Split the values
            train_codes, test_codes = train_test_split(transformed_codes, train_size=size, random_state=seed)
            train_codes, validation_codes = train_test_split(train_codes, train_size=size, random_state=seed)

            # Write each type of files
            for output_name, output_list in zip(["train", "validation", "test"], [train_codes, validation_codes, test_codes]):
                output = output_path + "/" + output_name + ".txt"
                write_to_file(output, [code for l in output_list for code in l])
        except ValueError as e:
            print("An error occurred when splitting the transformed codes probably due to an empty set of generated "
                  "mutants. Please consider checking the source file/directory you provided.")
            print(f"Original message of the error: {e}")

    else:
        # Get the name of the output file
        output = output_path + "/" + "results.txt"
        transformed_codes = [code for list_codes in transformed_codes for code in list_codes]
        write_to_file(output, transformed_codes)

# -------------------------------------------------------------------


if __name__ == "__main__":
    # Create the argument parser for the fault injection code
    parser = argparse.ArgumentParser(description=
                                     "Program allowing the generation of mutant code according to specific rules "
                                     "and associates a label which each type of transformation. The rules should be "
                                     "formatted as explained in the 'rules_grammar.txt' file. The randomness of the "
                                     "rules is specified according to the presence of the 'random' parameter in the "
                                     "arguments of this program. "
                                     "Every part of the program has been made to be compatible with the CESReS model "
                                     "based on BERT to highlight misconceptions in the student codes. The code is "
                                     "provided 'AS IT' and most errors you could get are most probably caused by a "
                                     "misconfiguration or possibly a bug in the current code. "
                                     "The label 'correct' has a specific meaning. Every mutation done on a code having "
                                     "this label will remove this previously associated label. "
                                     "See the rules files to have example of possible rules and how to format them.\n"
                                     "Author: Guillaume Steveny ; Year: 2023-2024")

    parser.add_argument("-t", "--train_split", action="store", type=int, default=None,
                        help="Ensure creating a train, a validation and a test file containing subset of the generated "
                             "mutants. The value to be passed to this argument is the percentage (expressed between 0 "
                             "and 100) to be use when splitting the train set and validation set from the test set.")

    parser.add_argument("-r", "--random", nargs="?", const=42, default=None,
                        help="Specifies if a random rule should be picked when generating mutants. This parameter is "
                             "also used when performing '->' rules where a random match is selected instead of the "
                             "first one if multiple are available. The value you can give to this parameter is a "
                             "specific seed you would like to use (not mandatory, the default seed is 42 to ensure "
                             "reproducibility of experiments)")

    parser.add_argument("-l", "--limit", action="store", type=int, default=-1,
                        help="Allows to specify the total number of mutants to be kept for each original code. By "
                             "default, the program will keep all the generated codes. Setting a value greater than "
                             "your number of rules, will result in the same behaviour.")

    parser.add_argument("-p", "--processes", action="store", type=int, default=10,
                        help="Specifies the number of processes to be used in the ThreadPool to operate on the input"
                             "files. A first loop will create the task asynchronously for the pool and a second will"
                             "recover the results. Using ctrl+c during the first loop will switch to the second."
                             "Breaking the second, automatically switch to the output writing."
                             "(default = 10)")

    parser.add_argument("-m", "--max_time", action="store", type=int, default=10,
                        help="Maximum allowed time (in seconds) for each rule to be applied on a code. When the "
                             "allowed time expires, the newt rule is tried until each rule has been tested. "
                             "(default = 10)")

    parser.add_argument("-n", "--number_files", action="store", type=int, default=-1,
                        help="Total number of files to be considered inside the source. By default this parameter is "
                             "ignored and all files are used (could lead to bad format of the progress bar in the "
                             "tgz way)")

    parser.add_argument("-f", "--filter", action="store", type=str, default=None,
                        help="File extension that should be considered when creating new mutants. If the source is a "
                             "directory, by default, only the '.py' files are selected, if a tgz archive is chosen "
                             "(or equivalently with the 'extract' parameter using the default tgz file) the default "
                             "file extension is the '.test' one created by Inginious.")

    parser.add_argument("-e", "--extract", action="store_true",
                        help="Specify if the extracting of a .tgz file should be done. This is a shortcut for putting "
                             "the ../output/submissions.tgz input as source parameter in the command line. Adding this "
                             "parameter while specifying a .tgz file as source has no additional effects.")

    parser.add_argument("-a", "--anonym", action="store_true",
                        help="Indicates if the name of each processed file should be transformed into a MD5 hash of "
                             "its original value to ensure anonym mutated submissions.")

    parser.add_argument("-s", "--source", action="store", type=str, default='.',
                        help="Source directory in which the original code that should be mutated can be found. These "
                             "should be saved in the '.py' format (other files are ignored except told differently by "
                             "using the filter argument). If rules cannot be "
                             "applied on a specific code, the modification is ignored to avoid mislabelling. "
                             "Putting a .tgz file for this parameter will go over the compressed .test file formatted "
                             "as yaml ones like Inginious output them. "
                             "(default = the current directory '.')")

    parser.add_argument("-o", "--output", action="store", type=str, default="output/modified_code",
                        help="The directory name in which the resulting 'txt' files should be stored after the "
                             "mutant generation. It is not mandatory for you to create it in advance, the program will "
                             "try to do it for you without suppressing the contained data if it already exists. The "
                             "created files depend on the t / train_split parameter. Without it, all the mutants are "
                             "in the 'results.txt' file, otherwise you will get the 'train.txt', 'validation.txt' and "
                             "'test.txt' files. A 'all_codes.txt' file will be also created for the resampling "
                             "procedure."
                             "(default = output/modified_code)")

    parser.add_argument("rule_file", action="store", type=str,
                        help="Path of the file containing the different rules. Bad formatted lines are ignored (which "
                             "allows you to write comments in plain text (as long as you do not use ' => ' or ' -> ' "
                             "and ' ; ') inside these. If a rule does not match for an specific code, it will be "
                             "ignored by the program and another rule will be tried.")

    # Get the current args given to the program
    args = parser.parse_args()

    # Creates the output dir (ignore the creation if it already exists)
    try_create_dir(args.output)

    # Object needed to modify the codes according to the rules specified in the rule_file argument
    comby = CombyTimeout(timeout=args.max_time)

    # Initiate the random seed if specified
    if args.random is not None:
        nr.seed(args.random)
        rd.seed(args.random)

    # Get the rules parsed according to the specific format selected
    # These are formatted as
    # LABEL COUNT
    # $   PATTERN (=>|->) REPLACEMENT
    # or
    # £   REDBARON_RULE
    # ;
    # This format is specified in the 'rules_grammar.txt' file
    # Here => means "replace all matches", -> means only a random one if -r else the first one
    # For additional details and structures information, checks the 'rules_grammar.txt' file

    # Please take care of the spaces in the specified format
    rules = parse_rules(args.rule_file, comby)

    # Get the result of precessing the files according to the rules
    results = process_all_files(args.source, rules)

    # Write the results '.txt' files
    to_cesres_file(results, args.output)
