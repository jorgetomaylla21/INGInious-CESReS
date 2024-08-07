# ============================================================================================
#
# GraphCodeBERT input generation
#
# Author: Guillaume Steveny
# Year: 2023 -- 2024
#
# Part of this code comes from the GitHub repository of
# Microsoft which created GraphCodeBERT.
# Link:
# https://github.com/microsoft/CodeBERT/blob/master/GraphCodeBERT/clonedetection/run.py#L91
#
# ============================================================================================

from typing import Any, Sequence

import torch
from transformers import PreTrainedTokenizer
from inspect4py.utils import extract_dataflow, DFG_python
import numpy as np
from tree_sitter import Language, Parser


# Code taken and adapted from https://github.com/microsoft/CodeBERT/blob/master/GraphCodeBERT/clonedetection/run.py#L91
# This function represents the input generation for GraphCodeBERT Microsoft's research team used for clone detection.
# We added the handling of null window length and tokenization
def convert_code_to_features(
        code: str,
        tokenizer: PreTrainedTokenizer,
        code_length: int = 256,
        dataflow_length: int = 256,
        language_library: str = "./my-language.so"
) -> dict[str, list[Any]]:
    """
    Convert a code inside a string into the features needed as input of the GraphCodeBERT model.

    Args:
        code: str containing the code to be transformed into its features'.
        tokenizer: PretrainedTokenizer from HuggingFace which transform the code into tokens.
        code_length: maximal number of tokens to be kept.
        dataflow_length: maximal number of variables of the DFG to keep.
        language_library: the name of the compiled file for parsing the code (.dll on Windows without the extension,
                          .so on Linux).

    Returns:
        a dict[str, list[Any]] containing for each feature name, the value of this feature.
    """
    # Create a tree-sitter parser for the compiled language
    LANGUAGE = Language(language_library, "python")
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, DFG_python]

    # Get the tokens and the dfg thanks to the inspect4py library
    code_tokens, dfg = extract_dataflow(code, parser, 'python')
    code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                   enumerate(code_tokens)]

    # Get the position span of each token
    ori2cur_pos = {-1: (0, 0)}
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))

    # Get the word pieces
    code_tokens = [y for x in code_tokens for y in x]

    # Truncate
    if code_length == 0:
        code_tokens = []
    else:
        code_tokens = code_tokens[:code_length + dataflow_length - 3 - min(len(dfg), dataflow_length)][:512 - 3]

    # The source tokens is CLS + code tokens + SEP
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)

    # Compute the position ids
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]

    # Truncate the DFG and add the values inside the tokens, position and ids
    if dataflow_length == 0:
        dfg = []
    else:
        dfg = dfg[:code_length + dataflow_length - len(source_tokens)]
    source_tokens += ["variable_" + x[0] + "_occurrence_" + str(e) for e,x in enumerate(dfg)]
    position_idx += [0 for x in dfg]
    source_ids += [tokenizer.unk_token_id for x in dfg]

    # Compute padding and add it to the features
    padding_length = code_length + dataflow_length - len(source_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    source_ids += [tokenizer.pad_token_id] * padding_length

    # Reindex
    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)

    # Will be useful to compute the attention mask
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]

    return {"input_tokens": source_tokens,
            "input_ids": source_ids,
            "position_idx": position_idx,
            "dfg_to_code": dfg_to_code,
            "dfg_to_dfg": dfg_to_dfg}


# Code taken and adapted from https://github.com/microsoft/CodeBERT/blob/master/GraphCodeBERT/clonedetection/run.py#L91
# This function represents the input generation for GraphCodeBERT Microsoft's research team used for clone detection.
# This only contains the part generating the code tokens.
def get_code_tokens(
        code: str,
        tokenizer: PreTrainedTokenizer,
        language_library: str = "./my-language.so"
) -> Sequence[str]:
    """
    Function transforming a code into a sequence of tokens to be used by the model or to map the most important
    tokens according to the interpretability part and their position inside the code text (used in the GUI).

    Args:
        code: the code string to be tokenized.
        tokenizer: the PreTrainedTokenizer to be used for tokenization.
        language_library: the name of the compiled file for parsing the code (.dll on Windows without the extension,
                          .so on Linux).

    Returns:
        a Sequence of strings where each entry is a token (preceded by \u0120 for token not being the first one).
    """
    # Create a tree-sitter parser for the compiled language of python
    LANGUAGE = Language(language_library, "python")
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, DFG_python]

    # Get the tokens and the dfg thanks to the inspect4py library
    code_tokens, _ = extract_dataflow(code, parser, 'python')
    code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                   enumerate(code_tokens)]

    # Get the word pieces
    code_tokens = [y for x in code_tokens for y in x]

    return code_tokens


# Code taken and adapted from https://github.com/microsoft/CodeBERT/blob/master/GraphCodeBERT/clonedetection/run.py#L91
def show_features(features: dict[str, list[Any]]) -> None:
    """
    Allows to show the features generated by the convert_code_to_features function

    Args:
        features: the output of convert_code_to_features function

    Returns:
        None
    """
    print("input_tokens: {}".format([x.replace('\u0120', '_') for x in features["input_tokens"]]))
    print("input_ids: {}".format(' '.join(map(str, features["input_ids"]))))
    print("position_idx: {}".format(features["position_idx"]))
    print("dfg_to_code: {}".format(' '.join(map(str, features["dfg_to_code"]))))
    print("dfg_to_dfg: {}".format(' '.join(map(str, features["dfg_to_dfg"]))))
    print()


# Code taken and adapted from https://github.com/microsoft/CodeBERT/blob/master/GraphCodeBERT/clonedetection/run.py#L91
# This function represents the input generation for GraphCodeBERT Microsoft's research team used for clone detection.
def input_from_features(
        features: dict[str, list[Any]],
        code_length: int = 256,
        dataflow_length: int = 256
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Function to transform the features from the convert_code_to_features function to the input needed by the
    GraphCodeBERT model.

    Args:
        features: the output of the convert_code_to_features function.
        code_length: the maximal number of code tokens to keep.
        dataflow_length: the maximal number of nodes in the DFG to keep.

    Returns:
        the Tensor of the input_ids, attention_mask and position_ids for the model.
    """
    # Get the features
    input_tokens = features["input_tokens"]
    input_ids = features["input_ids"]
    position_idx = features["position_idx"]
    dfg_to_code = features["dfg_to_code"]
    dfg_to_dfg = features["dfg_to_dfg"]

    # Initiate the attention mask
    attn_mask = np.zeros((code_length + dataflow_length, code_length + dataflow_length), dtype=bool)

    # Get information about the input
    node_index = sum([i > 1 for i in position_idx])
    max_length = sum([i != 1 for i in position_idx])

    # The tokens are all taken
    attn_mask[:node_index, :node_index] = True

    # Set to True for the input tokens
    for idx, i in enumerate(input_ids):
        if i in [0, 2]:
            attn_mask[idx, :max_length] = True

    # Encode the DFG relationships in the attention mask
    for idx, (a, b) in enumerate(dfg_to_code):
        if a < node_index and b < node_index:
            attn_mask[idx + node_index, a:b] = True
            attn_mask[a:b, idx + node_index] = True

    for idx, nodes in enumerate(dfg_to_dfg):
        for a in nodes:
            if a + node_index < len(position_idx):
                attn_mask[idx + node_index, a + node_index] = True

    # Return the constructed tensors
    return torch.Tensor(input_ids), torch.Tensor(position_idx), torch.Tensor(attn_mask)


def put_semicolon_last_indent(code: str) -> str:
    """
    Function to automatically add a semicolon at the end of the last instruction of each indented block.

    Args:
        code: the code to be modified to add the semicolon indication.

    Return:
        the modified code.
    """
    # Last indentation level
    last_indent = 0
    # The lines of the newly constructed code
    new_lines = []

    # For each line
    for line in code.split("\n"):
        # We only consider the non-empty lines
        if len(line) > 0:
            # The indentation is the number of \t or sequence of 4 spaces
            pos = 0
            while pos < len(line) and (line[pos] == " " or line[pos] == "\t"):
                pos += 1
            # NOTE: lines only composed of spaces could break this part
            indent = pos
            # The semicolon is added at the previous line if the current line is less indented
            if indent < last_indent:
                new_lines[-1] = new_lines[-1]+";"
            # Update the set of lines and indentation
            new_lines.append(line)
            last_indent = indent
    # Check if the last line should also have a ";"
    if len(new_lines[-1]) > 0 and new_lines[-1][-1] != ";":
        new_lines[-1] = new_lines[-1]+";"

    # The new code is the new lines
    return "\n".join(new_lines)
