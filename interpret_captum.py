# ===========================================================
#
# Interpreter - CESReS utilities for Captum
#
# Author: Guillaume Steveny
# Year: 2023 -- 2024
#
# This program has been developed following by taking
# inspiration from the official Captum tutorial:
# https://captum.ai/tutorials/Bert_SQUAD_Interpret
#
# ===========================================================

from __future__ import annotations

import warnings

import torch
from allennlp.data.fields import TensorField
from torch import Tensor
from transformers import RobertaModel
from captum.attr import LayerIntegratedGradients, LayerConductance, InternalInfluence, LayerFeatureAblation
from typing import Callable


class CaptumInterpreter:
    """
    Class representing a wrapper around the model and the interpreters inside the Captum library.
    Currently, it only supports LayerIntegratedGradients and LayerConductance.
    Warning: for LayerConductance, you can only use the 'bert_interpretable_layer' layer.

    Attributes:
        model: the CESReS model used for prediction.
        dataset_reader: the CESReS reader used to transform code into inputs.
        predictor: the CESReS predictor used for prediction.
        interpreter: the Captum interpreter object used to attribute score to the tokens.
        attribute_kwargs: the additional parameters to be given to each attribute call of the interpreter object.
        ref_id: the id of the reference index to create the baseline for IntegratedGradients-like interpreters = [PAD].
        cls_id: the id of the [CLS] token to create the baseline for IntegratedGradients-like interpreters.
        sep_id: the id of the [SEP] token to create the baseline for IntegratedGradients-like interpreters.

    Args:
        model: the CESReS model used for prediction.
        dataset_reader: the CESReS reader used to transform code into inputs.
        predictor: the CESReS predictor used for prediction.
        interpreter_name: a str to represent the type of Captum interpreter we want to use.
                          (default="LayerIntegratedGradients")
        layer: a str for the name of the layer we want to use. Could be 'bert_embeddings' or 'bert_interpretable_layer'.
               If None, this means that the interpreter is not a Layer version. (default = None)
        attribute_kwargs: a dict containing the additional arguments for the attribute function of the interpreter
                          (default={"n_steps":2, "internal_batch_size":1})
    """

    __slots__ = ["model", "dataset_reader", "predictor", "interpreter", "attribute_kwargs",
                 "ref_id", "cls_id", "sep_id"]

    # Possible Captum interpreters for the model
    interpreters = {
        "LayerIntegratedGradients": LayerIntegratedGradients,
        "LayerConductance": LayerConductance,
        # NOTE: this is also a possible choice: "InternalInfluence": InternalInfluence,
        # NOTE: this is also a possible choice: "LayerFeatureAblation": LayerFeatureAblation
    }

    def get_bert_interpretable_layer(self):
        """
        Method retrieving the embedder layer of the model for the LayerIntegratedGradients interpreter.

        Returns:
            the torch Module corresponding to the embedder layer of the model.
        """
        return self.predictor.get_interpretable_layer()

    def __init__(self,
                 model: "ClassificationEmbedderModel",
                 dataset_reader: "CodeReader",
                 predictor: "CodeClassifierPredictor",
                 interpreter_name: str = "LayerIntegratedGradients",
                 layer: str | None = None,
                 attribute_kwargs: dict | None = None
                 ):
        # Save the parameters used for prediction of a new example
        self.model = model
        self.dataset_reader = dataset_reader
        self.predictor = predictor

        # Get the reference token ids
        self.ref_id = self.dataset_reader.tokenizer.tokenizer.pad_token_id
        self.cls_id = self.dataset_reader.tokenizer.tokenizer.cls_token_id
        self.sep_id = self.dataset_reader.tokenizer.tokenizer.sep_token_id

        # Possible layers for the LayerIntegratedGradients
        layers = {
            "bert_interpretable_layer": self.get_bert_interpretable_layer,
            "bert_embeddings": lambda: [m.embeddings for m in self.model.modules() if isinstance(m, RobertaModel)][:1]
        }

        # Get the interpreter selected
        if interpreter_name not in self.interpreters:
            warnings.warn("Value warning, the interpreter you selected does not exist currently in this "
                          "implementation. "
                          "The default value LayerIntegratedGradients is selected. Please consider using one of "
                          f"{list(self.interpreters.keys())} the next time.")
            interpreter_name = "LayerIntegratedGradients"

        interpreter_class = self.interpreters[interpreter_name]

        # Init the interpreter
        if layer is not None:
            if layer not in layers:
                warnings.warn("Value warning, the layer you selected does not exist currently in this "
                              "implementation. "
                              "The default value bert_interpretable_layer is selected. Please consider using one of "
                              f"{list(layers.keys())} the next time.")
                layer = "bert_interpretable_layer"
            self.interpreter = interpreter_class(self.get_forward_function(), layers[layer]())
        else:
            self.interpreter = interpreter_class(self.get_forward_function())

        # Get the other args for using attribute
        self.attribute_kwargs = {
            "n_steps": 2,
            "internal_batch_size": 1
        }
        # The LayerFeatureAblation does not accept these parameters
        if interpreter_name == "LayerFeatureAblation":
            self.attribute_kwargs = {}
        # Update with the other args the config used
        if attribute_kwargs is not None:
            self.attribute_kwargs.update(attribute_kwargs)

    # Inspired by the official Captum tutorial: https://captum.ai/tutorials/Bert_SQUAD_Interpret
    def get_forward_function(self) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
        """
        Create a callable that will call the forward method of the model and only recover the probs of the
        prediction.

        Returns:
            a callable that return the probs outputted by the model when fed with the input_ids, the attention mask
            and the position_ids.
        """
        def forward_func(input_ids: Tensor, mask: Tensor, positions_ids: Tensor) -> Tensor:
            """
            Wrapper function around the forward method of the model to only return the probs outputted by the model.

            Args:
                input_ids: a Tensor associated with the ids of the tokenized version of the input code.
                mask: a Tensor containing the attention mask to be used inside the embedder.
                positions_ids: a Tensor containing the positional embedding of the input sequence.

            Returns:
                 a Tensor containing for each label, the confidence level at which this label is predicted.
            """
            output = self.model.forward(input_ids, mask, positions_ids)
            return output["probs"]
        return forward_func

    # Inspired by the official Captum tutorial: https://captum.ai/tutorials/Bert_SQUAD_Interpret
    def construct_baseline(
            self,
            input_ids: Tensor,
            position_ids: Tensor,
            device: str = "cpu"
    ) -> Tensor:
        """
        Function constructing the baseline for IntegratedGradients-like interpreters assuming the model to use
        GraphCodeBERT input representation, i.e. the code tokens have a position id that is greater than 1.
        This method create a tensor filled with ref_id and with [CLS] at the start and [SEP] at the [SEP] positions in
        the input.

        Args:
            input_ids: the GraphCodeBERT input ids generated by the functions in transform_code_to_df.py.
            position_ids: the GraphCodeBERT position ids generated by the functions in transform_code_to_df.py.
            device: the device on which the tensors should be handled (default = 'cpu').

        Returns:
            the baseline Tensor generated from the inputs.
        """
        # Fill the baseline with ref_id
        baseline = torch.ones_like(input_ids, device=device) * self.ref_id
        # Put [CLS] at the start of each input
        baseline[:, 0] = self.cls_id
        # Get the number of code tokens in each input
        code_length = [sum(position_ids[i] > 1) for i in range(len(position_ids))]
        # Set the [SEP] to the corresponding positions
        baseline[:, code_length] = self.sep_id
        return baseline

    def interpret(self,
                  code: str,
                  label_info: tuple[int, str, float],
                  limit: int = 10,
                  quiet: bool = False,
                  device: str = "cpu"
                  ) -> list[tuple[int, tuple[str, float]]]:
        """
        Method to get a score attribution to each token inside the source code. This interpretation focuses on a
        specific output label given by the label_info argument.

        Args:
            code: the source code from which we want to get the prediction and the score attribution.
            label_info: a tuple containing the index of the selected label, its textual name and the confidence level.
            limit: an int to indicate the maximal number of high scoring (in absolute value) tokens.
            quiet: a bool to indicate if the method should print the result.
            device: the device on which the tensor should be stored (default = "cpu").

        Returns:
            the score associated with each entry.
        """
        # Get the features created on this source code
        feat = self.dataset_reader.get_features(code)

        # Get the ids, mask and positions in the Instance created on the code
        inst: dict[str, TensorField] = self.dataset_reader.text_to_instance(code)
        input_ids = inst["input_ids"].tensor.unsqueeze(0).to(device)
        mask = inst["mask"].tensor.unsqueeze(0).to(device)
        positions_ids = inst["positions_ids"].tensor.unsqueeze(0).to(device)

        # Construct the baseline
        baseline = self.construct_baseline(input_ids, positions_ids, device)

        # Get the interpretation from Captum
        interpret = self.interpreter.attribute(
            inputs=input_ids,
            baselines=baseline,
            additional_forward_args=(mask, positions_ids),
            target=[label_info[0]],
            **self.attribute_kwargs
        )

        # Get the interpretation scores
        if isinstance(interpret, list):
            interpret = interpret[0]

        # Normalizes the attribution output
        # Following the official Captum tutorial: https://captum.ai/tutorials/Bert_SQUAD_Interpret
        attr = interpret.sum(dim=-1).squeeze(0)
        attr = attr / torch.norm(attr)

        # Get the token in a nice format
        f = [x.replace('\u0120', '_') for x in feat["input_tokens"]]

        # Construct the pairing between the positions, the tokens and their score
        inter_ids = list(zip(f, attr))
        inter_ids_pos = list(enumerate(inter_ids))

        # Sort the scores and limit to the maximal number of tokens to keep
        inter_sort = sorted(inter_ids_pos, key=lambda x: abs(x[1][1]), reverse=True)[:limit]

        # Print the results if the quiet argument is not set
        if not quiet:
            print(f"Scores for the label {label_info[1]} (confidence = {label_info[2]:.3f})")
            for t in inter_sort:
                print(f"\tToken n°{t[0]}: {t[1][0]} -> {t[1][1]}")

        # Return the constructed pairing
        return inter_sort
