# ===========================================================
#
# Mutation labelling program - Abstract rules
#
# Author: Guillaume Steveny
# Year: 2023 -- 2024
#
# ===========================================================

from __future__ import annotations
from abc import ABC, abstractmethod


class NoMatches(ValueError):
    """
    Class indicating the absence of a match for a specific pattern
    """
    def __init__(self):
        super().__init__("No match found for the current source code and specified rule")


class AbstractRule(ABC):
    """
    Abstract class representing a rule to be applied to a code and for which the generated mutant should be associated
    with a specific label

    Attributes:
        label: a string containing the label to be associated with the mutant after the modification
        expr:  the file line that was used to create this rule (if any, otherwise this is an empty string)

    Args:
        label: a string containing the label to be associated with the mutant after the modification
        expr:  the file line that was used to create this rule (if any, otherwise this is an empty string)
    """
    __slots__ = ["label", "expr"]

    def __init__(self, label: str, expr: str = ""):
        self.label = label
        self.expr = expr

    def construct_label(self, info: tuple[str, int, str], raise_correct: bool = False):
        """
        Method allowing to reconstruct a label associated with a code if this snippet was already tagged with a
        specific set of labels.

        Args:
            info: the information associated with this code snippet to be mutated
            raise_correct: bool to avoid mutating codes already labelled correct (raises NoMatches if it tries to do so)

        Returns:
            a string corresponding to the sorted set of labels associated with the code (the correct label is removed
            if the mutation created an error in the code)

        Raises
            NoMatches if the label is already assigned to the code snippet
        """
        # Create a set to represent the previous labels
        labels = set(info[2].split(" ; ")) if info[2] != '' else set()

        # Get the label(s) for this rule
        rule_labels = set(self.label.split(","))

        # Check if the label is not already associated with this code
        if labels.isdisjoint(rule_labels):
            # If correct is in label, these labels don't contain correct, so remove correct from labels
            if "correct" in labels:
                # If raise_correct is set, we do not allow already labelled 'correct' code to be mutated
                if raise_correct:
                    raise NoMatches()
                # Otherwise, we remove the correct label
                labels.remove("correct")
            # If no labels or everything except correct, we add the labels of the rule
            if len(labels) == 0 or "correct" not in rule_labels:
                labels.update(rule_labels)
        # However, we don't allow the same type of rule to be applied multiple times
        elif "correct" not in rule_labels:
            raise NoMatches()

        # The label is the association between the previous labels (sorted)
        label = " ; ".join(sorted(labels))

        return label

    @abstractmethod
    def apply(self, src: str, info: tuple[str, int, str]) -> tuple[str, int, str, str]:
        """
        Method to apply the rule specified by this object and get the result and corresponding label(s)

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
        pass

    def __str__(self):
        return self.expr

    def __repr__(self):
        return str(self)

