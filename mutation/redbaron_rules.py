# ===========================================================
#
# Mutation labelling program - RedBaron rules
#
# Author: Guillaume Steveny
# Year: 2023 -- 2024
#
# ===========================================================

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from random import Random
import string
from typing import Sequence

from mutation_rule import AbstractRule, NoMatches

from redbaron import RedBaron, NodeList, EndlNode, TryNode, NameNode, AssignmentNode, AtomtrailersNode, \
    UnitaryOperatorNode, DefArgumentNode, BinaryOperatorNode, IntNode, FloatNode, ForNode, IfNode, WhileNode, \
    WithNode, Node, IfelseblockNode, CallNode, LineProxyList, DotNode, DefNode, PrintNode


# Inspired by https://github.com/PyCQA/redbaron/issues/173#issuecomment-465711244
# We modified the node filtering
def backup_indentation(red) -> dict[Node, int]:
    """
    Function to create a dict structure containing the indentation level for a set of RedBaron nodes

    Args:
        red: the RedBaron object for which some indentation levels should be saved

    Returns:
        a dict where each key (Node) is associated with an int corresponding to the indentation of this node
    """
    # Output dictionary
    out: dict[Node, int] = {}
    # List of all the node types to be able to restore
    nodes = red.find_all(["except", "if", "else", "elif", "return", "assign", "print", "try", "while", "for"])
    # Each node is saved
    for node in nodes:
        out[node] = len(node.indentation)
    return out


# Inspired by https://github.com/PyCQA/redbaron/issues/173#issuecomment-465711244
# We modified the node filtering
def restore_indentation(red, indents: dict[Node, int], omit: set[Node] = set()) -> None:
    """
    Function to restore the indentation level of different nodes

    Args:
        red: the RedBaron object for which saved nodes should have their indentation restored
        indents: the dict outputted by backup_indentation where the indentation levels are saved
        omit: a set of RedBaron nodes to be omitted when restoring the indentation

    Returns:
        None
    """
    # Set of nodes for which the indentation could be retored
    nodes = red.find_all(["except", "if", "else", "elif", "return", "assign", "print", "try", "while", "for"])
    # For each node
    for node in nodes:
        # If the node should not be omitted
        if node not in omit and node in indents:
            # Get the indentation of the saved instance
            new_indentation = len(node.indentation)
            # Restore it thanks to the provided methods
            if new_indentation > indents[node]:
                node.decrease_indentation(new_indentation-indents[node])
            elif new_indentation < indents[node]:
                node.increase_indentation(indents[node]-new_indentation)


class RedBaronRule(AbstractRule, ABC):
    """
    Abstract class representing the different rules that could be constructed by using the RedBaron library

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """
    __slots__ = ["label", "random", "rd", "expr"]

    def __init__(self, label: str, random: bool, rd: Random | None, expr: str = ""):
        super().__init__(label, expr)
        self.random = random
        self.rd = rd

    @abstractmethod
    def operate(self, red: RedBaron) -> str:
        """
        Method implementing the behaviour of each rule to generate a mutant code. This is always called on the RedBaron
        AST of the source code, and it returns the modified code as a string

        Args:
            red: the RedBaron object construct on the source code

        Returns:
            a string representing the modified code

        Raises:
            any king of exception that should be considered as a non-matching source code, raising a NoMatches Exception
            in the 'apply' code where this method is called
        """
        pass

    def apply(self, src: str, info: tuple[str, int, str]) -> tuple[str, int, str, str]:
        # If something wrong happens in the modification, we suppose a NoMatch
        try:
            # Create the new label
            label = self.construct_label(info)

            # Get the AST representation of the code
            red = RedBaron(src)

            # Create the output tuple
            return info[0], info[1], self.operate(red), label

        except Exception as e:
            raise NoMatches()


class RemoveTry(RedBaronRule):
    """
    Mutation rule representing the removing of a Try-Except block inside a Python code.
    The generated code will contain the nested code block in place of the try block.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get the original code
        src = red.dumps()

        # Find all the node corresponding to a try-except block
        all_try = red.find_all("try")

        # Select a block (if any, randomly if it was specified)
        if len(all_try) <= 0:
            raise ValueError()
        selected: TryNode = all_try[0] if not self.random else self.rd.choice(all_try)

        # Get the string representing of the complete selected block
        patch = selected.dumps()

        # Get only a copy of the inner body of the selected try
        body_selected = selected.value.copy().data

        # Ensure not having an additional line feed to start this block
        if type(body_selected[0]) == EndlNode:
            body_selected = body_selected[1:]

        # Transform if to a list of nodes
        body_selected = NodeList(body_selected)

        # Ensure the body to be indented as its parent block
        body_selected.decrease_indentation(4)

        # Get the text representation of this block
        replace = body_selected.dumps()

        return src.replace(patch, replace)


class OtherVariable(RedBaronRule):
    """
    Mutation rule representing the replacement of a variable inside a Python code by another variable in the same
    snippet.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the variable used in the code
        all_vars: Sequence[NameNode] = red.find_all("name")

        # Select randomly a variable to be replaced
        to_replace = self.rd.choice(all_vars) if self.random else all_vars[0]

        # Select all the variables which are not the one to be replaced
        possible_selection: Sequence[NameNode] = red.find_all("name", value=lambda v: v != to_replace.value)
        selected_var = self.rd.choice(possible_selection) if self.random else possible_selection[0]

        # Change the value of the replaced by the replacement
        to_replace.value = selected_var.value

        return red.dumps()


class RandomVariable(RedBaronRule):
    """
    Mutation rule representing the replacement of a variable inside a Python code by another variable which is
    randomly generated and which is not already present in the snippet.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def generate_random_variable(self, size: int = 5, added: int = 1) -> str:
        """
        Function generating a random name of a variable containing only lowercase ascii letters.
        This generated string will be composed of size characters and an additional suffix of random chars composed
        of at most added components.

        Args:
            size:  an int to be the base size of the newly created variable name
            added: an int representing the maximal number of chars to add to the created variable name

        Returns:
            a string containing at least size lowercase ascii char and at most size + added

        Raises:
            NoMatches if the random component of this rule was not specified (impossible to generate randomly
            something without a random generator)
        """
        if self.random:
            selected_name = ""
            for i in range(size + self.rd.randint(0, added)):
                selected_name += self.rd.choice(string.ascii_lowercase)
            return selected_name
        else:
            raise NoMatches()

    def operate(self, red: RedBaron) -> str:
        # Get all the variable used in the code
        all_vars: Sequence[NameNode] = red.find_all("name")

        # Select randomly a variable to be replaced
        to_replace = self.rd.choice(all_vars) if self.random else all_vars[0]

        # Generate a random variable to replace (error if no random specified)
        selected_name = self.generate_random_variable(len(to_replace.value))

        # Get all the other variables and regenerate if the variable already exists
        var_set = {n.value for n in all_vars}
        while selected_name in var_set:
            selected_name = self.generate_random_variable(len(to_replace.value))

        # Replace the variable by the randomly generated one
        to_replace.value = selected_name

        return red.dumps()


class ValueChange(RedBaronRule):
    """
    Mutation rule representing the modification of a numerical value inside a Python code. The expr argument can
    contain additional information to modify the behaviour of this rule:
    £   value_change      => modify randomly an int by doing a random increment (between -2 and +2 except 0)
    £   value_change i V  => modify randomly an int by doing +V
    £   value_change f V  => modify randomly a float by doing +V
    £   value_change a V  => modify randomly an int or a float by doing +V
    V is an optional parameter, if not specified, the increment is selected randomly between -2 and +2 except 0

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
        mode:   str[] specifying the list of type to consider
        values: int[] containing the possible increment value

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def __init__(self, label: str, random: bool, rd: Random | None, expr: str = ""):
        super().__init__(label, random, rd, expr)

        # Modes correspondences
        modes = {"i": ["int"], "f": ["float"], "a": ["int", "float"]}

        # Set the mode and value to default
        self.mode = ["int"]
        self.values = [1, -2, -1, 2]

        # Check if there is additional parameter to the rule
        params = expr.split(" ")

        # If there is a second parameter (specify the mode)
        if len(params) > 1:
            self.mode = modes[params[1]]

        # If there is a third parameter (change the increment)
        if len(params) > 2:
            if "." in params[2]:
                self.values = [float(params[2])]
            else:
                self.values = [int(params[2])]

    def operate(self, red: RedBaron) -> str:
        # Get all the nodes corresponding to the mode specified
        all_vals = []
        for mode in self.mode:
            for node in red.find_all(mode):
                all_vals.append(node)

        # Choose a random value if specified, otherwise take the first one
        selected_val = self.rd.choice(all_vals) if self.random else all_vals[0]

        # Convert the value in the node to its correct correspondant
        converted_value = float(selected_val.value) if "." in selected_val.value else int(selected_val.value)

        # Modify the corresponding field to correspond to the increment
        if type(selected_val.parent) == UnitaryOperatorNode:
            increment = (self.rd.choice(self.values) if self.random else self.values[0])
            new_value = str(-converted_value + increment)
            selected_val.parent.replace(new_value)
        else:
            selected_val.value = str(converted_value + (self.rd.choice(self.values) if self.random
                                                        else self.values[0]))

        return red.dumps()


class ChangeSign(RedBaronRule):
    """
    Mutation rule representing the sign change of a numerical value inside a Python code. The expr argument can
    contain additional information to modify the behaviour of this rule:
    £   change_sign    => invert randomly an int
    £   change_sign i  => invert randomly an int
    £   change_sign f  => invert randomly a float
    £   change_sign a  => invert randomly an int or a float

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
        mode:   str[] specifying the list of type to consider

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def __init__(self, label: str, random: bool, rd: Random | None, expr: str = ""):
        super().__init__(label, random, rd, expr)

        # Modes correspondences
        modes = {"i": ["int"], "f": ["float"], "a": ["int", "float"]}

        # Set the mode and value to default
        self.mode = ["int"]

        # Check if there is additional parameter to the rule
        params = expr.split(" ")

        # If there is a second parameter (specify the mode)
        if len(params) > 1:
            self.mode = modes[params[1]]

    def operate(self, red: RedBaron) -> str:
        # Get all the nodes corresponding to the mode specified
        all_vals = []
        for mode in self.mode:
            for node in red.find_all(mode):
                if node.value != "0":
                    all_vals.append(node)

        # Choose a random value if specified, otherwise take the first one
        selected_val = self.rd.choice(all_vals) if self.random else all_vals[0]

        # Check the parent of the value to see if it is positive or not
        if type(selected_val.parent) == UnitaryOperatorNode:
            if selected_val.parent.value == "-":
                selected_val.parent.replace(selected_val.parent.dumps()[1:])
            else:
                selected_val.parent.replace("-"+selected_val.parent.dumps())
        else:
            selected_val.replace("-"+selected_val.dumps())

        return red.dumps()


class RemoveAssignWhile(RedBaronRule):
    """
    Mutation rule representing the modification of an assignment inside a Python code linked with all variables used
    in the condition of a while loop. The expr argument can contain additional information to modify the behaviour
    of this rule:
    £   remove_assign_while    => suppress all the value update inside first-level body of a while for all vars in the
                                  condition
    £   remove_assign_while i  => suppress one of the value update in the first parent level of a while for on var in
                                  the condition

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
        init:   bool to indicate if the removal operation is done inside the body or at the first parent level

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def __init__(self, label: str, random: bool, rd: Random | None, expr: str = ""):
        super().__init__(label, random, rd, expr)

        # By default, we remove the update inside the while
        # If this attribute is True, we will remove the first assign concerning a variable in the parent
        self.init = False

        # Check the presence of additional optional parameter
        params = expr.split(" ")

        # Update consequently the attribute
        if len(params) > 1:
            self.init = params[1] == "i"

    def operate(self, red: RedBaron) -> str:
        # Find all the while nodes in the code
        all_whiles = red.find_all("while")

        # Select a while to be mutated
        selected_while = self.rd.choice(all_whiles) if self.random else all_whiles[0]

        # Get the variables used in the test of the while
        test_vars = selected_while.test.find_all("name")
        set_vars = {t.value for t in test_vars}

        # Backup the indentation levels to restore them
        d = backup_indentation(red)

        # Check which sort of assign should be removed
        if not self.init:
            # Get all the assign nodes in the while loop
            all_assign = selected_while.value.find_all("assign")

            # Filter the assign to correspond to an update to a variable in the test
            kept_assign = [a for a in all_assign if a.target.value in set_vars]

            # Remove the update lines
            for k in kept_assign:
                k.parent.remove(k)

        else:
            # Get the all the assign
            all_assign = [n for n in selected_while.parent if type(n) == AssignmentNode]

            # Keep only the assigns matching the condition in loop
            kept_assign = [a for a in all_assign if a.target.value in set_vars]

            # Select the first matching assign and deletes it
            kept_assign[0].parent.remove(kept_assign[0])

        # Restore the indentation of different nodes
        restore_indentation(red, d)

        return red.dumps()


class WithToOpen(RedBaronRule):
    """
    Mutation rule representing the replacement of a with block inside a Python code.
    The expr argument can contain additional information to modify the behaviour of this rule:
    £   with_to_open    => replace a with block with its content following the as expressed as assignment
    £   with_to_open c  => same as the normal rule but the '.close()' are added at the end of the replaced block

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
        close:  bool to indicate if the '.close()' operation should be added at the end of the replacement

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def __init__(self, label: str, random: bool, rd: Random | None, expr: str = ""):
        super().__init__(label, random, rd, expr)

        # By default, we don't add the .close() method
        self.close = False

        # Check f there is additional optional parameters
        params = expr.split(" ")

        # If one is found, check if it impacts the mutation (other values are ignored)
        if len(params) > 1:
            self.close = params[1] == "c"

    def operate(self, red: RedBaron) -> str:
        # Reconstruct the source code
        src = red.dumps()

        # Get all the with nodes corresponding to access of a ressource
        all_with = red.find_all("with")

        # Select a with inside all the possible choices
        selected_with = self.rd.choice(all_with) if self.random else all_with[0]

        # Get the sequence to be replaced
        pattern = selected_with.dumps()

        # Get all the replaceable context
        contexts = [c for c in selected_with.contexts if c.as_ is not None]

        # Get the ressource variables to be closed
        to_close = []

        # Construct the replacement
        new_open = []
        for context in contexts:
            new_line = context.as_.dumps() + " = " + context.value.dumps()
            if self.close:
                to_close.append(context.as_.dumps() + ".close()")
            new_open.append(new_line)

        # Get the body of the with
        with_body = selected_with.value

        # Insert the open inside the body
        for opens in new_open:
            with_body.insert(0, opens)

        # Check if the close operation should be done and add them to the body of the with
        if self.close:
            for to_be_closed in to_close:
                with_body.append(to_be_closed)

        # Create a list of nodes to represent this body
        with_node = NodeList(with_body)

        # The indentation should be removed
        with_node.decrease_indentation(4)

        # Get the code patch of the with_node
        replace = with_node[0].dumps()
        if True or not self.close:
            for l in with_node[1:]:
                replace += "\n" + selected_with.indentation + l.dumps()
        if self.close:
            replace += "\n" + selected_with.indentation

        return src.replace(pattern, replace)


class RemoveReturn(RedBaronRule):
    """
    Mutation rule representing the removal of a return statement inside a Python code.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the return nodes
        all_return = red.find_all("return")

        # Select the return to be suppressed
        selected_return = self.rd.choice(all_return) if self.random else all_return[0]

        # Replace the return by an empty indentation
        selected_return.replace(selected_return.indentation)

        return red.dumps()


class ForRange(RedBaronRule):
    """
    Mutation rule representing the modification of a for statement inside a Python code.
    It modifies a "for VAR in SEQ" (where SEQ is not a call to range) to "for VAR in range(SEQ)"

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the for nodes
        all_for = red.find_all("for")

        # Filter the one we can replace
        filter_for = [f for f in all_for if f.target.find("name", value="range") is None]

        # Select a for
        selected_for = self.rd.choice(filter_for) if self.random else filter_for[0]

        # Modifies the target of for
        selected_for.target.replace(f"range({selected_for.target.dumps()})")

        return red.dumps()


class RemoveNestedLoop(RedBaronRule):
    """
    Mutation rule representing the deletion of a loop inside a Python code.
    The expr argument can contain additional information to modify the behaviour of this rule:
    £   remove_nested_loop    => replace the loop by its body
    £   remove_nested_loop d  => suppress the loop (replace by an empty hole)

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
        keep:   boolean to indicate if the body should be conserved or not

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def __init__(self, label: str, random: bool, rd: Random | None, expr: str = ""):
        super().__init__(label, random, rd, expr)

        # Keep the body of the loop
        self.keep = True

        # Check if there is additional optional parameters
        params = expr.split(" ")

        # If one is found, check if we should delete the body of the loop
        if len(params) > 1:
            self.keep = params[1] != "d"

    def operate(self, red: RedBaron) -> str:
        # Get all fors
        # NOTE: when removing a for, it creates unassigned variables inside the body which replaces the loop
        all_for = red.find_all("for")

        # Get all the whiles
        all_whiles = red.find_all("while")

        # Combine all loops
        all_loops = []
        all_loops.extend(all_for)
        all_loops.extend(all_whiles)

        # Select a loop to be removed
        selected_loop = self.rd.choice(all_loops) if self.random else all_loops[0]

        # Replace the loop according to this rule parameter
        if self.keep:
            nodelist_copy = NodeList(selected_loop.value.copy())
            nodelist_copy.decrease_indentation(4)
            for node in nodelist_copy:
                if type(node) != EndlNode:
                    selected_loop.insert_before(node)
            selected_loop.parent.remove(selected_loop)
        else:
            selected_loop.replace("\n" + (selected_loop.next_recursive.indentation
                                          if selected_loop.next_recursive is not None else ''))

        return red.dumps()


class RemoveIfElseBlock(RedBaronRule):
    """
    Mutation rule representing the replacement of an if-else block inside a Python code. This block is replaced by
    the content of the if block (or else block according to the additional parameter)
    The expr argument can contain additional information to modify the behaviour of this rule:
    £   remove_if_else   => replace the complete block by the content of the if
    £   remove_if_else e => replace the complete block by the content of the else (if any)

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
        keep_else:  bool indicating if the else is used to replace the complete block

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def __init__(self, label: str, random: bool, rd: Random | None, expr: str = ""):
        super().__init__(label, random, rd, expr)

        # By default, we keep the if not the else
        self.keep_else = False

        # Check if there is additional optional parameters
        params = expr.split(" ")

        # If one is found, check if it corresponds to "keeping else"
        if len(params) > 1:
            self.keep_else = params[1] == "e"

    def operate(self, red: RedBaron) -> str:
        # Get all the if blocks
        all_blocks = red.find_all("ifelseblock")

        # Select a block
        selected_block = self.rd.choice(all_blocks) if self.random else all_blocks[0]

        # Keep a node according to the attribute keep_else
        kept_node = selected_block.find("if" if not self.keep_else else "else")

        # Replace the complete if-else block by the content of the kept node
        nodelist_copy = NodeList(kept_node.value.copy())
        nodelist_copy.decrease_indentation(4)
        for node in nodelist_copy:
            if type(node) != EndlNode:
                selected_block.insert_before(node)
        selected_block.parent.remove(selected_block)

        return red.dumps()


class CompletelyRemoveCdtBlock(RedBaronRule):
    """
    Mutation rule to completely remove a statement conditional block inside a Python code.
    The expr argument can contain additional information to modify the behaviour of this rule:
    £   suppress_if       => suppress the complete block of if-elif-else
    £   suppress_if if    => suppress only the "if" part of the complete block
    £   suppress_if else  => suppress only the "else" part of the complete block
    £   suppress_if elif  => suppress only the "elif" part of the complete block

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
        type_sub_node:  str representing the abstract name of the type of nodes to be deleted

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def __init__(self, label: str, random: bool, rd: Random | None, expr: str = ""):
        super().__init__(label, random, rd, expr)

        # All the type of nodes to be suppressed
        selectors = {"if", "elif", "else", "ifelseblock"}

        # By default, we suppress the complete block
        self.type_sub_node = "ifelseblock"

        # Check if there is additional optional parameters
        params = expr.split(" ")

        # If one is found, adapt the selector accordingly (if not know, keep the default one)
        if len(params) > 1:
            if params[1] in selectors:
                self.type_sub_node = params[1]

    def operate(self, red: RedBaron) -> str:
        # Get all the nodes that could be suppressed
        all_nodes = red.find_all(self.type_sub_node)

        # Select a node to be deleted
        selected_node = self.rd.choice(all_nodes) if self.random else all_nodes[0]

        # Replace this node by an empty one
        if self.type_sub_node == "ifelseblock":
            selected_node.insert_before("")
            selected_node.parent.remove(selected_node)
        else:
            selected_node.replace("\n" + (selected_node.next_recursive.indentation
                                          if selected_node.next_recursive is not None else ''))

        return red.dumps()


class RevertSlice(RedBaronRule):
    """
    Mutation rule representing the modification of a slice inside a Python code.
    The expr argument can contain additional information to modify the behaviour of this rule:
    £   revert_slice    => put the lower bound in the upper bound and vice-versa
    £   revert_slice l  => change the sign of the lower value when inverting
    £   revert_slice u  => change the sign of the upper value when inverting
    £   revert_slice a  => change the sign of the all bounds when inverting
    The sign change is random if the random parameter is activated which means that generating a change happens
    with a probability of 0.5 in this configuration.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
        sign:   set[str] to indicate which value should have their sign changed

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def __init__(self, label: str, random: bool, rd: Random | None, expr: str = ""):
        super().__init__(label, random, rd, expr)

        # Change sign modes
        signs = {"l": {"lower"}, "u": {"upper"}, "a": {"lower", "upper"}}

        # By default, we don't change sign
        self.sign = set()

        # Check if there is additional optional parameters
        params = expr.split(" ")

        # If one is found, check if it impacts the mutation
        if len(params) > 1:
            self.sign = signs[params[1]]

    def operate(self, red: RedBaron) -> str:
        # Get all the slices
        all_slices = red.find_all("slice")

        # Filter the slices to only keep the non-empty one
        filter_slices = [s for s in all_slices if s.lower is not None or s.upper is not None]

        # Select a slice
        selected_slice = self.rd.choice(filter_slices) if self.random else filter_slices[0]

        # Get the lower and higher value
        lower = selected_slice.lower.dumps() if selected_slice.lower is not None else ""
        upper = selected_slice.upper.dumps() if selected_slice.upper is not None else ""

        # Modifies the bounds if it should be done
        # The sign is changed with a probability of 0.5 if the random parameter is set
        if lower != "" and "lower" in self.sign:
            if (self.random and self.rd.randint(0, 1) == 1) or (not self.random):
                if lower[0] == "-":
                    lower = lower[1:]
                else:
                    lower = "-" + lower
        if upper != "" and "upper" in self.sign:
            if (self.random and self.rd.randint(0, 1) == 1) or (not self.random):
                if upper[0] == "-":
                    upper = upper[1:]
                else:
                    upper = "-" + upper

        # Modify the value of the slice
        selected_slice.upper = lower
        selected_slice.lower = upper

        return red.dumps()


class SuppressPartSlice(RedBaronRule):
    """
    Mutation rule representing the removal of a part of a slice inside a Python code.
    The expr argument can contain additional information to modify the behaviour of this rule:
    £   suppress_part_slice    => suppress all the components of the slice (lower, upper and step)
    £   suppress_part_slice V  => suppress the components of the slice specified by V
    V is a sequence of characters indicating the chosen components. The possible characters are l (for lower), u (for
    upper) and s (for step).
    If the random parameter was activated, the choice of the parts to remove is also random

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
        value:  list[str] to indicate which components should be removed (or randomly selected for this)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def __init__(self, label: str, random: bool, rd: Random | None, expr: str = ""):
        super().__init__(label, random, rd, expr)

        # Change value modes
        values = {"l": "lower", "u": "upper", "s": "step"}

        # By default, we suppress all values
        self.value = list(values.values())

        # Check if there is additional optional parameters
        params = expr.split(" ")

        # If one is found, check if it impacts the mutation
        if len(params) > 1:
            self.value = [values[letter] for letter in params[1] if letter in values]

    def operate(self, red: RedBaron) -> str:
        # Get all the slices
        all_slices = red.find_all("slice")

        # Select the part that should be removed
        # This selection is random if this was specified when creating this rule
        selected_values = self.rd.sample(self.value, self.rd.randint(1, len(self.value))) \
            if self.random else self.value

        # Check the presence of attributes
        def checking(s):
            for v in selected_values:
                if getattr(s, v, None) is None:
                    return False
            return True

        # Filter the slices to only keep the non-empty one
        filter_slices = [s for s in all_slices if checking(s)]

        # Select a slice
        selected_slice = self.rd.choice(filter_slices) if self.random else filter_slices[0]

        # For each value that is selected, suppress its value
        for v in selected_values:
            setattr(selected_slice, v, '')

        return red.dumps()


class RemoveArgFromCall(RedBaronRule):
    """
    Mutation rule representing the removal of an argument in a function call inside a Python code.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the calls found in the program
        all_calls = red.find_all("call")

        # Filter the call to have at least one arg to be suppressed
        filter_calls = [c for c in all_calls if len(c.value) > 0]

        # Select a call to be processed
        selected_call = self.rd.choice(filter_calls) if self.random else filter_calls[0]

        # Select an arg to be suppressed
        selected_arg = self.rd.choice(selected_call.value) if self.random else selected_call.value[0]

        # Remove the arg from the argument list
        selected_call.value.remove(selected_arg)

        return red.dumps()


class RemoveCall(RedBaronRule):
    """
    Mutation rule representing the removal of a call operation inside a Python code.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the calls found in the program
        all_calls = red.find_all("call")

        # Select a call to be suppressed
        selected_call = self.rd.choice(all_calls) if self.random else all_calls[0]

        # Remove this call from its parent node
        selected_call.parent.remove(selected_call)

        return red.dumps()


class RemoveSelf(RedBaronRule):
    """
    Mutation rule representing the removal of a self variable inside a Python code.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the self variables
        all_self = red.find_all("name", value="self")

        # Select a self
        selected_self = self.rd.choice(all_self) if self.random else all_self[0]

        # Suppress this from its parent
        if type(selected_self.parent) == DefArgumentNode:
            # There is a bug in redbaron that suppress the space between def and name if the arguments become empty
            if len(selected_self.parent.parent.arguments) == 1:
                selected_self.parent.parent.name = " " + selected_self.parent.parent.name
            # We remove the parent form the arguments of the parent of this parent
            selected_self.parent.parent.arguments.remove(selected_self.parent)
        else:
            selected_self.parent.remove(selected_self)

        return red.dumps()


class ListToTuple(RedBaronRule):
    """
    Mutation rule representing the transformation of a list creation in its tuple equivalent inside a Python code.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the list definition nodes
        all_lists = red.find_all("list")

        # Select a list to be transformed
        selected_list = self.rd.choice(all_lists) if self.random else all_lists[0]

        # Get the representation of the list to transform it in tuple
        list_dump = selected_list.dumps()

        # Transform this list into a tuple
        tuple_dump = f"({list_dump[1:-1]})"

        # Replace the list node by the tuple representation
        selected_list.replace(tuple_dump)

        return red.dumps()


class ChangeComparison(RedBaronRule):
    """
    Mutation rule representing the change of a binary comparison operation inside a Python code.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
        operators: a set[str] containing all the comparison operators that are allowed for replacement

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def __init__(self, label: str, random: bool, rd: Random | None, expr: str = ""):
        super().__init__(label, random, rd, expr)

        self.operators = {"==", "!=", "<=", ">=", "<", ">"}

    def operate(self, red: RedBaron) -> str:
        # Get all the comparison operators
        all_operators = red.find_all("comparison_operator")

        # Select an operator to be modified
        selected_operator = self.rd.choice(all_operators) if self.random else all_operators[0]

        # Reconstruct the operator
        opera = selected_operator.first + selected_operator.second

        # Remove the operator from the operators
        restricted_operators = list(self.operators.difference({opera}))

        # Select a new operator
        new_operator = self.rd.choice(restricted_operators) if self.random else restricted_operators[0]

        # Replace the operator
        selected_operator.first = new_operator[0]
        selected_operator.second = new_operator[1] if len(new_operator) > 1 else ''

        return red.dumps()


class Identity(RedBaronRule):
    """
    Class representing the identity transformation of a source code. The outputted value of "apply" is its input
    value labeled with the corresponding label.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not - ignored
        rd:     the Random object used to make random choices (None if choice are not randomized) - ignored
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not - ignored
        rd:     the Random object used to make random choices (None if choice are not randomized) - ignored
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """
    def operate(self, red: RedBaron) -> str:
        pass

    def apply(self, src: str, info: tuple[str, int, str]) -> tuple[str, int, str, str]:
        return info[0], info[1], src, self.construct_label(info)


class MisplacedReturn(RedBaronRule):
    """
    Mutation rule representing the misplacement of a return inside a Python code.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
        operators: a set[str] containing all the comparison operators that are allowed for replacement

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the return nodes
        all_returns = red.find_all("return")

        # Filter the returns so that there is mode than a single instruction in its parent body
        filter_return = [r for r in all_returns if len(r.parent) > 1]

        # Select the return to be misplaced
        selected_return = self.rd.choice(filter_return) if self.random else filter_return[0]

        # Get all the node different inside the parent
        filter_body = [n for n in selected_return.parent if n is not selected_return]

        # Select the instruction to be used to misplace
        selected_instruction = self.rd.choice(filter_body) if self.random else filter_body[0]

        # Backup the indentation levels to restore them
        d = backup_indentation(red)

        # Suppress the return from its parent
        selected_return.parent.remove(selected_return)

        # Place the return before the selected instruction
        selected_instruction.insert_before(selected_return)

        # Restore the indentation of different nodes
        restore_indentation(red, d)

        return red.dumps()


class AssignToComparison(RedBaronRule):
    """
    Mutation rule representing the modification of a "=" into a "==" inside a Python code.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
        operators: a set[str] containing all the comparison operators that are allowed for replacement

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the assign node
        all_assign = red.find_all("assign")

        # We filter the assign to be simple assign (no operator)
        filter_assign = [a for a in all_assign if a.operator == '']

        # Selecting an assign
        selected_assign = self.rd.choice(filter_assign) if self.random else filter_assign[0]

        # Replace if by a comparison
        selected_assign.replace(f"{selected_assign.target.dumps()} == {selected_assign.value.dumps()}")

        return red.dumps()


class ComparisonToAssign(RedBaronRule):
    """
    Mutation rule representing the modification of a "==" into a "=" inside a Python code.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
        operators: a set[str] containing all the comparison operators that are allowed for replacement

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the comparison operators
        all_operators = red.find_all("comparison_operator")

        # Filter the operators to be only for "=="
        filter_operators = [o for o in all_operators if o.first + o.second == "=="]

        # Select an operator to be replaced
        selected_operator = self.rd.choice(filter_operators) if self.random else filter_operators[0]

        # Replace the parent of this operator by an assign
        selected_operator.parent.replace(f"{selected_operator.parent.first} = {selected_operator.parent.second}")

        return red.dumps()


class ChangeOperandOrder(RedBaronRule):
    """
    Mutation rule representing the operand inversion inside a Python code. If we have an expression "x OPE y", the
    rule change it to "y OPE' x" where OPE' is the modified OPE (invert the '>'/'<' if the operation is a comparison,
    otherwise OPE' = OPE)

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
        to_invert: a dict[str, str] containing how comparison operators should be modified

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def __init__(self, label: str, random: bool, rd: Random | None, expr: str = ""):
        super().__init__(label, random, rd, expr)

        self.to_invert = {
            ">=": "<=",
            ">": "<",
            "<=": ">=",
            "<": ">"
        }

    def invert_operator(self, operator) -> str:
        """
        Method receiving an operator and reverting it if necessary

        Args:
            operator: the operator to be reverted (could be a RedBaron node or a string)

        Returns:
            the operator if no correspondence are found in the to_invert dict, otherwise its inverted version
        """
        ope = operator.dumps() if type(operator) != str else operator
        return self.to_invert.get(ope, ope)

    def operate(self, red: RedBaron) -> str:
        # All the possible operators
        operators = ["comparison", "binary_operator", "boolean_operator"]

        # Get all the operators
        all_operators = []
        for ope in operators:
            all_ope_in_this_type = red.find_all(ope)
            filter_ope_type = [o for o in all_ope_in_this_type if o.value not in {"/", "%", "//", "-"}]
            all_operators.extend(filter_ope_type)

        # Select an operator
        selected_operator = self.rd.choice(all_operators) if self.random else all_operators[0]

        # Exchange the two parts
        tmp = selected_operator.first
        selected_operator.first = selected_operator.second
        selected_operator.second = tmp

        # Change the operator if needed
        selected_operator.value = self.invert_operator(selected_operator.value)

        return red.dumps()


class ClearCondensedAssign(RedBaronRule):
    """
    Mutation rule to replace each AugAssign (+=, *=, ....) by a simple assign (=) inside a Python code.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the assign
        all_assign = red.find_all("assign")

        # Filter the assign to only keep the one which are operation-assigns
        filter_assign = [a for a in all_assign if a.operator != '']

        # Select the assign to be updated
        selected_assign = self.rd.choice(filter_assign) if self.random else filter_assign[0]

        # Modify the operator
        selected_assign.operator = ''

        return red.dumps()


class UnravelCondensedAssign(RedBaronRule):
    """
    Mutation rule to transform a condensed AugAssign (+=, *=, ...) by its simplest form inside a Python code.
    For example, the instruction "i += 1" becomes "i = i + 1".

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the assign
        all_assign = red.find_all("assign")

        # Filter the assign to only keep the one which are operation-assigns
        filter_assign = [a for a in all_assign if a.operator != '']

        # Select the assign to be updated
        selected_assign = self.rd.choice(filter_assign) if self.random else filter_assign[0]

        # Modify the operator
        var = selected_assign.target.dumps()
        ope = selected_assign.operator
        val = selected_assign.value.dumps()
        selected_assign.replace(f"{var} = {var} {ope} {val}")

        return red.dumps()


class RavelAssign(RedBaronRule):
    """
    Mutation rule to transform a classical assign by its condensed form inside a Python code.
    For example, the instruction "i = i + 1" becomes "i += 1". This only works if the target value is present in a
    binary expression in the right-hand side.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the assign
        all_assign = red.find_all("assign")

        # Keep all the assign that are not condensed ones
        filter_assign = [a for a in all_assign if a.operator == '']

        # The only possible manner to ravel an assign, is that its value is an operation where one operand
        # is the variable
        kept_assign = []
        for assign in filter_assign:
            if type(assign.value) == BinaryOperatorNode:
                var = assign.target.dumps()
                if assign.value.first.dumps() == var:
                    ope = assign.value.value
                    kept_assign.append((assign, f"{var} {ope}= {assign.value.second.dumps()}"))
                elif assign.value.second.dumps() == var and assign.value.value not in {"/", "%", "//"}:
                    ope = assign.value.value
                    kept_assign.append((assign, f"{var} {ope}= {assign.value.first.dumps()}"))

        # Select a kept assign
        selected_assign = self.rd.choice(kept_assign) if self.random else kept_assign[0]

        # Modify the selected assign
        selected_assign[0].replace(selected_assign[1])

        return red.dumps()


class AddReturnToInit(RedBaronRule):
    """
    Mutation rule to add a return instruction in the __init__ method definition of a class inside a Python code.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the function/method definitions
        all_def = red.find_all("def")

        # Filter the def to be only the __init__ one
        filter_def = [d for d in all_def if d.name == "__init__"]

        # Select an init method
        selected_init = self.rd.choice(filter_def) if self.random else filter_def[0]

        # Get the indentation of the body of the init method
        init_indent = selected_init.indentation + (" "*4)

        # Select what the return should be
        selected_return = self.rd.choice(["return self", f"return {selected_init.parent.name}"]) \
            if self.random else "return self"

        # Create a back of the indentation
        backup = backup_indentation(red)

        # Add a return at the end
        selected_init.value.append(selected_return)

        # Adapt the indentation of the added return
        added_return = selected_init.value[-1]
        added_return.increase_indentation(len(init_indent)-len(added_return.indentation))

        # Restore the maybe lost indents
        restore_indentation(red, backup)

        return red.dumps()


class OutOfBoundRange(RedBaronRule):
    """
    Mutation rule to add a "+1" in an upper bound of a call to the range function inside a Python code.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the for loops
        all_for = red.find_all("for")

        # Filter all the fors to keep the one calling range
        filter_for = [f for f in all_for if f.target.find("name", value="range") is not None]

        # Select the for to be modified
        selected_for = self.rd.choice(filter_for) if self.random else filter_for[0]

        # Get the range name in the trailer
        selected_range = selected_for.target.find("name", value="range")

        # Get the call node after the range name
        range_call = selected_range.next

        # Retrieve the value of the upper bound
        if len(range_call.value) == 1:
            upper_bound = range_call.value[0]
        else:
            upper_bound = range_call.value[1]

        # Check if this value is a number or not and modify it accordingly
        if type(upper_bound.value) == IntNode:
            upper_bound.replace(str(int(upper_bound.dumps()) + 1))
        elif type(upper_bound.value) == FloatNode:
            upper_bound.replace(str(float(upper_bound.dumps()) + 1))
        else:
            upper_bound.replace(f"{upper_bound.dumps()} + 1")

        return red.dumps()


class DivisionChange(RedBaronRule):
    """
    Mutation rule to transform a "/" (resp. "//") by a "//" (resp. "/") inside a Python code.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the operation
        all_operator = red.find_all("binary_operator")

        # Filter the operators to keep only the / and the //
        filter_operator = [o for o in all_operator if o.value == "/" or o.value == "//"]

        # Select an operator randomly (if possible)
        selected_operator = self.rd.choice(filter_operator) if self.random else filter_operator[0]

        # Modify the operator
        if len(selected_operator.value) == 1:
            selected_operator.value += "/"
        else:
            selected_operator.value = "/"

        return red.dumps()


class PrintBeforeReturn(RedBaronRule):
    """
    Mutation rule to add print instruction before a return inside a Python code.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the return instructions
        all_returns = red.find_all("return")

        # Select a return from this list
        selected_return = self.rd.choice(all_returns) if self.random else all_returns[0]

        # Backup the indentation levels to restore them
        d = backup_indentation(red)

        # Put a print before the return
        selected_return.insert_before(f"print({selected_return.value.dumps()})")

        # Restore the indentation of different nodes
        restore_indentation(red, d)

        return red.dumps()


class ReturnInIndentedBlock(RedBaronRule):
    """
    Mutation rule to indent a return with respect to its previous block inside a Python code.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the return instructions
        all_returns = red.find_all("return")

        # Filter the return to only keep the ones after an indented block
        filter_return = [r for r in all_returns if type(r.previous) in {ForNode, IfelseblockNode, WhileNode, WithNode}]

        # Select a return
        selected_return = self.rd.choice(filter_return) if self.random else filter_return[0]

        # Make the return too much indented
        selected_return.increase_indentation(4)

        return red.dumps()


class BadOpenMode(RedBaronRule):
    """
    Mutation rule to modify the mode selected in an open function call inside a Python code.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the name representing the open function
        all_open = red.find_all("name", value="open")

        # Create the set of modes
        modes = {"w", "r"}

        # Filter the opens to correspond to open calls
        filter_open = [o for o in all_open if type(o.next) == CallNode and o.index_on_parent == 0 and
                       len(o.next.value) > 1 and o.next.value[1].dumps()[1:-1] in modes]

        # Select an open
        selected_open = self.rd.choice(filter_open) if self.random else filter_open[0]

        # Get the mode of the open
        open_mode = selected_open.next.value[1].dumps()
        mode = open_mode[1:-1]

        # Remove this mode from the modes
        modes.remove(mode)

        # Select a mode
        selected_mode = self.rd.choice(list(modes)) if self.random else list(modes)[0]

        # Put the new mode in the open
        selected_open.next.value[1].replace(f"{open_mode[0]}{selected_mode}{open_mode[-1]}")

        return red.dumps()


class NoClose(RedBaronRule):
    """
    Mutation rule to remove a close method call inside a Python code.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the name representing the close method
        all_close = red.find_all("name", value="close")

        # Check that the close is followed by a call, that the statement could be removed and
        # that the previous node is a dotnode
        filter_close = [c for c in all_close if type(c.next) == CallNode and
                        getattr(c.parent.parent, "remove", None) is not None and
                        type(c.previous) == DotNode]

        # Select a close
        selected_close = self.rd.choice(filter_close) if self.random else filter_close[0]

        # Get a backup of the indentations
        d = backup_indentation(red)

        # Remove the parent from its parent
        selected_close.parent.parent.remove(selected_close.parent)

        # Restore the indentation
        restore_indentation(red, d)

        return red.dumps()


class MissExcept(RedBaronRule):
    """
    Mutation rule to remove an exception catch in a try-except block inside a Python code.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the except nodes
        all_except = red.find_all("except")

        # Select an except
        selected_except = self.rd.choice(all_except) if self.random else all_except[0]

        # Create a backup of the indentations
        d = backup_indentation(red)

        # Remove this except from its parent
        selected_except.parent.excepts.remove(selected_except)

        # Restore the indentation
        restore_indentation(red, d)

        return red.dumps()


class ReplaceAllOccurrenceOfAVariable(RedBaronRule):
    """
    Mutation rule to replace each occurrence of a variable by a different variable name inside a Python code.
    The expr argument can contain additional information to modify the behaviour of this rule:
    £   replace_all_var    => uses the default list as first choice and then tries to create random names
    £   replace_all_var r  => ignores the default list and directly tries with random names.
    This rule is written thanks to RedBaron instead of the symtable or ast modules since these require executing
    the input code, which could be dangerous on non-checked entries.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
        default: a boolean to specify if the default list of possible names (a, b, c, x, y, z, w, var, variable)
                 should be considered

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def __init__(self, label: str, random: bool, rd: Random | None, expr: str = ""):
        super().__init__(label, random, rd, expr)

        self.default = True

        params = expr.split(" ")

        if len(params) > 1:
            self.default = not (params[1] == "r")

    def generate_random_variable(self, size: int = 5, added: int = 1) -> str:
        """
        Function generating a random name of a variable containing only lowercase ascii letters.
        This generated string will be composed of size characters and an additional suffix of random chars composed
        of at most added components.

        Args:
            size:  an int to be the base size of the newly created variable name
            added: an int representing the maximal number of chars to add to the created variable name

        Returns:
            a string containing at least size lowercase ascii char and at most size + added

        Raises:
            NoMatches if the random component of this rule was not specified (impossible to generate randomly
            something without a random generator)
        """
        if self.random:
            selected_name = ""
            for i in range(size + self.rd.randint(0, added)):
                selected_name += self.rd.choice(string.ascii_lowercase)
            return selected_name
        else:
            raise NoMatches()

    def operate(self, red: RedBaron) -> str:
        # Import the builtins
        import builtins

        # Get all the name nodes
        all_names = red.find_all("name")

        # Filter the name to be not a builtin function nor an object attribute
        filter_names = [n for n in all_names if getattr(builtins, n.value, None) is None and
                        type(n.previous) != DotNode and n.value not in {"True", "False", "None"}]

        # Create the set of variable names
        variable_names = set()
        for name in filter_names:
            variable_names.add(name.value)

        # Select a filtered node
        selected_name = self.rd.choice(filter_names) if self.random else filter_names[0]

        # List of default possible choices
        possible_names = ["a", "b", "c", "x", "y", "z", "w", "var", "variable"] if self.default else []
        # If the choice is random, the list is shuffled
        if self.random:
            self.rd.shuffle(possible_names)
        # Go over each possible replacement
        pos = 0
        selected_value = selected_name.value
        # While there is a correspondance for this value
        while selected_value in variable_names and pos < len(possible_names):
            # Test the next value
            selected_value = possible_names[pos]
            pos += 1
        # If we arrived at the end of possible choices
        if pos == len(possible_names) and selected_value in variable_names:
            # Try to create random new names
            tries = 0
            while selected_value in variable_names and tries < 26:
                # If the random mode is set, create a random name
                # Otherwise, create a name with multiple ascii chars
                selected_value = self.generate_random_variable(len(selected_name.value), 1) if self.random else \
                    string.ascii_lowercase[tries]
                tries += 1
            if tries == 26 and selected_value in variable_names:
                raise NoMatches()

        # Transform each occurrence of this selected name into the new selected one
        initial_value = selected_name.value
        for name in filter_names:
            if name.value == initial_value:
                name.value = selected_value

        # If the name refers to a function, rename the function accordingly
        all_def = red.find_all("def", name=initial_value)
        for defs in all_def:
            defs.name = selected_value

        # Return the update code
        return red.dumps()


class RenameAllDef(RedBaronRule):
    """
    Mutation rule to replace function name by a default name inside a Python code.
    The functions are renamed in their appearance order. The first becomes 'function_1' and the last 'function_n'
    with n the number of def statements. Each variable with the same name as the functions is modified accordingly.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the def
        all_defs = red.find_all("def")

        # Def name dict
        def_name_dict = {}
        for defs in all_defs:
            if defs.name not in def_name_dict:
                def_name_dict[defs.name] = []
            def_name_dict[defs.name].append(defs)

        # For each selected name
        function_number = 1
        for def_name, nodes in def_name_dict.items():
            # Create the new name
            new_name = f"function_{function_number}"
            # Go over the defs with the same name
            for defs in nodes:
                defs.name = new_name
            # Get all the names that could correspond to this function
            all_function_names = red.find_all("name", value=def_name)
            # Rename accordingly
            for name in all_function_names:
                name.value = new_name
            function_number += 1

        return red.dumps()


class RenameAllVarsDummy(RedBaronRule):
    """
    Mutation rule to replace all variables by dummy names inside a Python code.
    The variables are renamed in their appearance order. The first becomes 'var_1' and the last 'var_n'
    with n the number of unique names. Each function with the same name as the variables is modified accordingly.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        import builtins

        # Get all the names
        all_names = red.find_all("name")

        # Filter the names to not be attributes or builtin names
        filter_names = [n for n in all_names if getattr(builtins, n.value, None) is None and
                        type(n.previous) != DotNode and n.value not in {"True", "False", "None"}]

        # Var name dict
        var_name_dict = {}
        for var in filter_names:
            if var.value not in var_name_dict:
                var_name_dict[var.value] = []
            var_name_dict[var.value].append(var)

        # For each selected name
        variable_number = 1
        for var_name, nodes in var_name_dict.items():
            # Create the new name
            new_name = f"var_{variable_number}"
            # Go over the names with the same name
            for var in nodes:
                var.value = new_name
            # Get all the names that could correspond to a function
            all_function_names = red.find_all("def", name=var_name)
            # Rename accordingly
            for defs in all_function_names:
                defs.name = new_name
            variable_number += 1

        return red.dumps()


class RemoveCommentsAndDocstrings(RedBaronRule):
    """
    Mutation rule to remove all comments and docstrings inside a Python code.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Backup the indentations
        backup = backup_indentation(red)

        # Get all the comments
        all_comments = red.find_all("comment")

        # Remove all comments
        for comment in all_comments:
            try:
                comment.parent.remove(comment)
            except Exception as _:
                pass

        # Get all docstrings
        all_strings = red.find_all("string")
        all_docstrings = [s for s in all_strings if type(s.parent) == DefNode and s.index_on_parent == 0]

        # Remove all docstrings
        for docstring in all_docstrings:
            docstring.parent.remove(docstring)

        # Restore the possibly lost indentation
        restore_indentation(red, backup)

        # Get the initial code
        code: str = red.dumps()

        # Replace all comments by nothing
        new_code = re.sub('#.*\n', '\n', code)

        return new_code


class RemoveParenthesis(RedBaronRule):
    """
    Mutation rule to remove one instance of associative parentheses inside a Python code.
    Example: (1 + 2) * 3 becomes 1 + 2 * 3.

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the associative parentheses
        all_parentheses = red.find_all("associative_parenthesis")

        # Filter the parentheses to be not part of a print instruction
        filter_parentheses = [p for p in all_parentheses if type(p.parent) != PrintNode]

        # Select a random instance
        selected_parentheses = self.rd.choice(filter_parentheses) if self.random else filter_parentheses[0]

        # Replace the node by its child
        selected_parentheses.replace(selected_parentheses.value)

        return red.dumps()


class HardcodeArg(RedBaronRule):
    """
    Mutation rule to add an assignment of a function arg to an integer as first instruction of the function definition
    inside a Python code. The value is randomly selected between 0 and 20 if the rule is random, otherwise 0 is always
    selected.
    Example:
        def fun(n):
            return n
        becomes
        def fun(n):
            n = INT_VALUE
            return n

    Attributes:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label:  a string containing the label to be associated with the mutant after the modification
        random: a boolean value indicating if random choice should be done or not
        rd:     the Random object used to make random choices (None if choice are not randomized)
        expr:   the file line that was used to create this rule (if any, otherwise this is an empty string)
    """

    def operate(self, red: RedBaron) -> str:
        # Get all the function definitions
        all_defs = red.find_all("def")

        # Filter the def to have at least one arg
        filter_defs = [d for d in all_defs if
                       len([a for a in d.arguments if type(a) == DefArgumentNode
                            and a.value is None and str(a.target) != "self"]) > 0]

        # Select a def
        selected_def = self.rd.choice(filter_defs) if self.random else filter_defs[0]

        # Select an argument in the list of nodes
        filter_args = [a for a in selected_def.arguments if
                       type(a) == DefArgumentNode and a.value is None and str(a.target) != "self"]
        selected_arg = self.rd.choice(filter_args) if self.random else filter_args[0]

        # Safe the indentation level
        backup = backup_indentation(red)

        # Put the arg to a value at the top of the body of the function
        selected_def.value[0].insert_before(f"{selected_arg.target} = {self.rd.randint(0, 20) if self.random else 0}")

        # Restore indentation
        restore_indentation(red, backup)

        return red.dumps()


possible_redbaron_rules = {
    "remove_try": RemoveTry,
    "other_variable": OtherVariable,
    "random_variable": RandomVariable,
    "value_change": ValueChange,
    "remove_assign_while": RemoveAssignWhile,
    "with_to_open": WithToOpen,
    "remove_return": RemoveReturn,
    "for_range": ForRange,
    "revert_slice": RevertSlice,
    "suppress_part_slice": SuppressPartSlice,
    "remove_arg_call": RemoveArgFromCall,
    "remove_call": RemoveCall,
    "change_sign": ChangeSign,
    "remove_nested_loop": RemoveNestedLoop,
    "remove_if_else": RemoveIfElseBlock,
    "suppress_if": CompletelyRemoveCdtBlock,
    "list_to_tuple": ListToTuple,
    "change_comparison": ChangeComparison,
    "identity": Identity,
    "remove_self": RemoveSelf,
    "misplace_return": MisplacedReturn,
    "assign_to_comparison": AssignToComparison,
    "comparison_to_assign": ComparisonToAssign,
    "change_operand_order": ChangeOperandOrder,
    "clear_operation_assign": ClearCondensedAssign,
    "unravel_operation_assign": UnravelCondensedAssign,
    "ravel_assign": RavelAssign,
    "return_in_init": AddReturnToInit,
    "out_of_bounds_range": OutOfBoundRange,
    "division_change": DivisionChange,
    "print_before_return": PrintBeforeReturn,
    "indent_return": ReturnInIndentedBlock,
    "bad_open_mode": BadOpenMode,
    "miss_close": NoClose,
    "miss_except": MissExcept,
    "replace_all_var": ReplaceAllOccurrenceOfAVariable,
    "rename_all_def": RenameAllDef,
    "rename_all_vars_dummy": RenameAllVarsDummy,
    "remove_comments_and_docstrings": RemoveCommentsAndDocstrings,
    "remove_parenthesis": RemoveParenthesis,
    "hardcode_arg": HardcodeArg
}

if __name__ == "__main__":
    print("This command line tool allows you to manually test the mutation rules.")
    print("Typing '?' shows all the rules name, while '?' followed by the name of the rule print the help.")
    print("Type '$$$' to quit.")

    # For errors without crashing
    import warnings

    # The help for all
    pattern_help = re.compile("\?")
    # The help for a particular rule
    pattern_rule_help = re.compile("\? *([A-Za-z_]+)")
    # Selecting a rule
    pattern_rule = re.compile("([A-Za-z_]+)")

    # Keep track of the user inputs
    current_input = ""
    # When the user types '$$$', it exits the tool
    while current_input != "$$$":
        # Ask the user its command
        current_input = input("> ")
        # Checks the possible matches
        match_rule_help = pattern_rule_help.match(current_input)
        match_help = pattern_help.match(current_input)
        match_rule = pattern_rule.match(current_input)
        # If the user asked for help on a specific rule
        if match_rule_help is not None:
            # Get the rule
            rule = match_rule_help.group(1)
            # If it does not exist, warn the user
            if rule not in possible_redbaron_rules:
                warnings.warn(f"{rule} does not exist, should be a value in {list(possible_redbaron_rules.keys())}.")
            else:
                # Otherwise, show the help
                help(possible_redbaron_rules[rule])
        # If the user asked for global help
        elif match_help is not None:
            print(f"Choose between {list(possible_redbaron_rules.keys())}.")
        # If the user specified a rule
        elif match_rule is not None:
            # Get the rule name
            rule = match_rule.group(0)
            # If the rule does not exist, warn the user
            if rule not in possible_redbaron_rules:
                warnings.warn(f"{rule} does not exist, should be a value in {list(possible_redbaron_rules.keys())}.")
            else:
                # Otherwise, tries to apply the rule
                try:
                    # Ask the code to the user
                    print("Write you code hereunder: ($$$ to quit)")
                    current_code = ""
                    complete_code = ""
                    # Wait for $$$ to stop
                    while current_code != "$$$":
                        current_code = input("\t> ")
                        if current_code != "$$$":
                            complete_code += current_code+"\n"
                    # Should the rule comes with additional parameters
                    additional_params = input("Additional params: ")
                    # Ask the user for randomness
                    random = input("Should the mutation be random y/n: ")
                    random_bool = random in {"y", "yes"}
                    # Get the results
                    result = possible_redbaron_rules[rule]("LABEL", random_bool, Random(), f"x {additional_params}")\
                        .apply(complete_code, ("NAME", 0, "PREVIOUS_LABELS"))
                    # Show the mutated code to the user
                    print("\nResult:")
                    print(result[2])
                # Each error is warned to the user
                except Exception as e:
                    warnings.warn(str(e))
                    print()
    # Exit the program at '$$$' reception
    exit()
