# ==================================================================
#
# CESReS classification model program - Main program
#
# Author: Guillaume Steveny
# Year: 2023 -- 2024
#
# The development of this code was made by following the
# official AllenNLP (the library we used) tutorial.
# Every method or function that is inspired from it, before
# we adapted these for our task, is referenced as "inspired by".
# We added all the parameters handling and the adaptation of the
# model (using pre-trained model, the classification head, the
# command line predictions, the GUI connection and the
# documentation).
# 
# Tutorial URL:
# https://guide.allennlp.org/training-and-prediction#4
#
# ==================================================================

from __future__ import annotations

import json
import warnings

import yaml

import os

import tempfile

import asyncio
from asyncio import StreamWriter, StreamReader

from typing import Iterable, Dict, Tuple, List, Any, Sequence

import numpy as np
import torch

import transformers
from transformers import AutoModel

from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance, Vocabulary, DataLoader
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.fields import LabelField, ArrayField, MultiLabelField, TensorField
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.interpret.saliency_interpreters import IntegratedGradient
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders.bert_pooler import BertPooler
from allennlp.predictors import Predictor
from allennlp.training import Trainer, GradientDescentTrainer
from allennlp.training.metrics import CategoricalAccuracy, F1MultiLabelMeasure
from allennlp.training.optimizers import HuggingfaceAdamWOptimizer
from allennlp.training.util import evaluate

from interpret_captum import CaptumInterpreter
from transform_code_to_df import convert_code_to_features, show_features, input_from_features, get_code_tokens

from collections import OrderedDict

# NOTE: you can add codecarbon with
# from codecarbon import track_emissions, OfflineEmissionsTracker

# ======================================================================================================================

# Types of encoders the user can specify in the configuration file
encoders = {
    "cls_label": lambda arg, kwargs: select_cls_embedding,
    "bert_pooler": lambda arg, kwargs: BertPooler(*arg, **kwargs)
}

# Types of accuracy the user can specify in the configuration file
accuracies = {
    "categorical_accuracy": lambda arg, kwargs: CategoricalAccuracy()
}

# Type of possible activation functions
activations = {
    "gelu": lambda arg, kwargs: torch.nn.GELU(),
    "leaky_relu": lambda arg, kwargs: torch.nn.LeakyReLU(),
    "relu": lambda arg, kwargs: torch.nn.ReLU()
}


def construct_sequential_head(number: int, hidden_sizes: int | list[int], num_labels: int, activation=None, norm=False):
    """
    Construct a sequential classification head with multiple dense layers

    Args:
        number: the number of dense layers to use before the classification layer
        hidden_sizes: an integer or list of integer. If this value is a single integer, it is supposed to be the
                      number of hidden units for each of the layers. If this is a list of integers, this should be
                      composed of ('number' + 1) values to specify a number of hidden units for each layer.
        num_labels: the number of labels for the last dense (classification) layer
        activation: parameter to indicate which type of activation function should be use between the layers
        norm: whether to use the Batch Normalization after the activation function

    Returns:
        a Sequential block composed of `number` denser layer + 1 classification layer
    """
    order = []

    if activation is None:
        activation = {"name": "gelu"}
    act_name = activation['name']

    if type(hidden_sizes) == list:
        assert len(hidden_sizes) + 1 == number, "You specified different hidden sizes but the number of layers is " \
                                                f"not coherent. Number of layers: {number}, number of values: " \
                                                f"{len(hidden_sizes)}. So you should have given {number + 1} values " \
                                                f"to the configuration."
        for i in range(number):
            order.append((f"classification_{i+1}", torch.nn.Linear(hidden_sizes[i], hidden_sizes[i+1])))
            order.append((f"{act_name}_{i+1}", create_parameter("activation", **activation)))
            if norm:
                order.append((f"norm_{i+1}", torch.nn.BatchNorm1d(hidden_sizes[i+1])))
        order += [("classification_output", torch.nn.Linear(hidden_sizes[-1], num_labels))]
    else:
        hidden_size = hidden_sizes
        for i in range(number):
            order.append((f"classification_{i+1}", torch.nn.Linear(hidden_size, hidden_size)))
            order.append((f"{act_name}_{i+1}", create_parameter("activation", **activation)))
            if norm:
                order.append((f"norm_{i+1}", torch.nn.BatchNorm1d(hidden_size)))
        order += [("classification_output", torch.nn.Linear(hidden_size, num_labels))]

    return torch.nn.Sequential(OrderedDict(order))


# Types of classifiers the user can specify in the configuration file
classifiers = {
    "simple": lambda arg, kwargs: torch.nn.Linear(*arg, **kwargs),
    "mult_dense": lambda arg, kwargs: construct_sequential_head(*arg, **kwargs)
}

# Type of losses that are usable in the configuration
losses = {
    "cross_entropy": lambda arg, kwargs: torch.nn.functional.cross_entropy,
    "multilabel_soft_margin_loss": lambda arg, kwargs: torch.nn.functional.multilabel_soft_margin_loss
}

# Types of parameters to be dynamically chosen by the used with its configuration file
param_types = {
    "encoder": encoders,
    "classification_head": classifiers,
    "accuracy": accuracies,
    "activation": activations,
    "loss": losses
}


class WrongParameter(Exception):
    """
    Class to specify a problem in the configuration value the user specified
    """
    def __init__(self, value, listing):
        super().__init__(f"Wrong parameter {value}.\nShould be one in the {listing} list.")


def create_parameter(param_type, name, arg=[], kwargs={}):
    """
    Function to create a specific parameter from the configuration parameters

    Args:
        param_type: a str corresponding to the parameter type (encoder, classification_hed or accuracy)
        name: a str corresponding to the name of the chosen value of the parameter
        arg: a list of arguments to be given to this parameter (ordered)
        kwargs: a dict of keywords arguments to be given to this parameter

    Returns:
        the constructed parameter for the model
    """
    # If the parameter is not a recognized type, raise exception
    if param_type not in param_types:
        raise WrongParameter(param_type, list(param_types.keys()))

    # If the selected value does not exist for this parameter
    if name not in param_types[param_type]:
        raise WrongParameter(param_type, list(param_types[param_type].keys()))

    # Return the parameter we could construct
    return param_types[param_type][name](arg, kwargs)


# ======================================================================================================================


# This class was written by using the AllenNLP official tutorial as inspiration.
# The tutorial can be found at: https://guide.allennlp.org/training-and-prediction#4
# The source code of this tutorial is: https://github.com/allenai/allennlp-guide/blob/master/quick_start/predict.py
# We modified the instance retrieving and tensor generation.
class CodeReader(DatasetReader):
    """
    Class representing a Dataset Reader able to recover the code snippets inside a txt file and the label associated
    with each of these.

    Attributes:
        snippet_splitter: a string representing the "code" used to split the different examples in the input file
        label_splitter:   a string representing the "code" used to split a snippet and the label associated with it
        multi_labels:     a string representing the splitter used to distinguish the different labels associated
                          with a code when performing a multi_label classification (None implies single label)
        part_graph: Sequence[int] containing information about the number of tokens to have in the code part and in the
                    graph part of the input
        tokenizer: PreTrainedTransformerTokenizer representing the tokenizer used to transform a code snippet
                   into a Token sequence to put inside the embedder
        indexer:   PreTrainedTransformerIndexer representing the indexer associated with the tokenizer specified
        debug:     bool representing the fact to show the features that are created by this object

    Args:
        huggingface_model: str representing the name of the pretrained model from huggingface
                           you want to use to tokenize the input data (should be a feature-extraction model and
                           the same that the embedder you want to use)
                           (default = GraphCodeBERT-py model from Enoch)
        snippet_splitter:  str representing the "code" used to split the different examples in the input file
                           (default = "\n$$$\n")
        label_splitter:    str representing the "code" used to split a snipper and the label associated with it
                           (default = " $x$ ")
        multi_labels:      str representing the separator used when multiple labels are associated with each code.
                           If this parameter is unspecified, the classification is supposed single class
                           (default = None)
        part_graph:        Sequence[int] composed of maximum two numbers and corresponding to the number of tokens to
                           keep for each part of the input. The first number is the code token count while the second
                           is the number of tokens in the graph (dfg) part of the input. This number can be zero if
                           the model is CodeBERT.
                           (default = (256, 256))
        compiled_language: str to indicate the path to the compiled library containing the parsing information for
                           tree-sitter when creating the tokenized instances. For windows, the file should have a
                           dll extension (without putting the extension in the path). On linux, this file should have a
                           so extension (here specified in the parameter).
                           (default = './my_language.so')
        kwargs_tokenizer:  Dict containing the additional args you want to put inside the tokenizer
                           (default = {max_length: 512})
        kwargs_indexer:    Dict containing the additional args you want to put inside the indexer
                           (default = {})
        debug: bool representing the fact to show the features that are created by this object
    """
    __slots__ = ["tokenizer", "indexer", "snippet_splitter", "label_splitter", "multi_labels", "part_graph",
                 "compiled_language", "debug"]

    def __init__(self,
                 huggingface_model: str = "Enoch/graphcodebert-py",
                 snippet_splitter: str = "\n$$$\n",
                 label_splitter: str = " $x$ ",
                 multi_labels: str | None = None,
                 part_graph: Sequence[int] = (256, 256),
                 compiled_language: str = "./my-language.so",
                 kwargs_tokenizer: Dict[str, Any] = None,
                 kwargs_indexer: Dict[str, Any] = None,
                 debug: bool = False):
        super().__init__()

        # Set a default value for the additional parameters of the tokenizer and the indexer
        if kwargs_tokenizer is None:
            kwargs_tokenizer = {"max_length": 512}

        if kwargs_indexer is None:
            kwargs_indexer = {}

        # Adding information to parse the entries in the input file
        self.snippet_splitter = snippet_splitter
        self.label_splitter = label_splitter
        self.multi_labels = multi_labels

        # Adding information about the part used after tokenization
        self.part_graph = part_graph[:2]

        # Add the information about the compiled language
        self.compiled_language = compiled_language

        # Creating the attributes of the object
        self.tokenizer = PretrainedTransformerTokenizer(huggingface_model, **kwargs_tokenizer)
        self.indexer = {'tokens': PretrainedTransformerIndexer(huggingface_model, **kwargs_indexer)}

        # Save the debug variable
        self.debug = debug

    # -------------------------------

    # Inspired by the official AllenNLP tutorial: https://guide.allennlp.org/training-and-prediction#4
    def text_to_instance(self, text: str, label: str = None) -> Instance:
        """
        Method to transform a string entry (text) into an AllenNLP Instance possibly associated with a label.

        Args:
            text:  str representing the text to be tokenized and on which the Instance should be created.
            label: str representing a possible label to be associated with the text entry.

        Returns:
            an Instance object containing the text tokenized and the label.
        """
        # Tokenizing the entry
        features = convert_code_to_features(text, self.tokenizer.tokenizer, *self.part_graph,
                                            language_library=self.compiled_language)
        if self.debug:
            show_features(features)

        # Get the input for the model
        ids, position, mask = input_from_features(features, *self.part_graph)

        # Transform it in Field for AllenNLP library
        ids_field = ArrayField(ids)
        mask_field = ArrayField(mask)
        position_field = ArrayField(position)

        # Create the fields for the instance (the text tokenized and indexed)
        fields: dict[str, TensorField | LabelField | MultiLabelField]
        fields = {'input_ids': ids_field, "mask": mask_field, "positions_ids": position_field}

        # If a label is associated with this code, add a LabelField to the Instance
        if label:
            if self.multi_labels:
                fields['label'] = MultiLabelField(label.split(self.multi_labels))
            else:
                fields['label'] = LabelField(label)

        return Instance(fields)

    # -------------------------------

    def get_features(self, text: str) -> dict[str, list[any]]:
        """
        Method to transform a text entry into the features generated for the GraphCodeBERT model.

        Args:
            text: str representing the text to be tokenized.

        Returns:
            a dict[str, List[Any]] containing for each feature name, the value of this feature.
        """
        return convert_code_to_features(text, self.tokenizer.tokenizer, *self.part_graph,
                                        language_library=self.compiled_language)

    # -------------------------------

    # Inspired by the official AllenNLP tutorial: https://guide.allennlp.org/training-and-prediction#4
    def _read(self, file_path) -> Iterable[Instance]:
        # Inherited method from the base DatasetReader class

        # Read the provided file
        with open(file_path, "r") as file:
            text = file.read()

        # We split the entry according to the specified code
        examples = text.split(self.snippet_splitter)[:-1]

        # For each create example
        for example in examples:
            # Split the text - label examples
            try:
                text, label = example.strip().split(self.label_splitter)
            # Avoid problems with empty codes
            except ValueError:
                continue

            # Yield a created Instance for this example
            yield self.text_to_instance(text, label)

# ======================================================================================================================


def select_cls_embedding(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Function allowing to recover the CLS token embedding from a tensor of dimension [batch_size, tokens, embedding_size].
    The CLS token should be the first one at position 0.

    Args:
        embeddings: a Tensor of dim [batch_size, tokens, embedding_size] where each component corresponds to the
                    embedding of a particular token in the input (and this for each component of the batch).

    Returns:
        a Tensor of dim [batch_size, 1, embedding_size] where the kept component corresponds to the embedding of the
        CLS token.
    """
    return embeddings[:, 0, :]

# ======================================================================================================================


# This class was written by using the AllenNLP official tutorial as inspiration.
# The tutorial can be found at: https://guide.allennlp.org/training-and-prediction#4
# The source code of this tutorial is: https://github.com/allenai/allennlp-guide/blob/master/quick_start/predict.py
# We modified the architecture and its initialization.
class ClassificationEmbedderModel(Model):
    """
    Class representing a model to classify input codes according to a particular pretrained embedder.

    Attributes:
        embedder:   PreTrainedTransformerEmbedder constructed from a model of the huggingface library which allows
                    to transform a input tokenized code into a Tensor of dimension [batch_size, tokens, embedding_size]
                    (default = GraphCodeBERT-py model from Enoch)
        encoder:    Callable[[Tensor], Tensor] allowing to transform the output of the embedder into a single embedding
                    for each component of the current batch
                    (default = bert_pooler)
        classifier: Linear layer for classifying according to the embedding
        accuracy:   Metric to compute the score of the classification. This argument could be omitted (set to None)
                    and nothing will be computed for this
                    (default = None)
        loss: Callable[[Tensor, Tensor] Tensor] allowing to compute the loss of the classification
              (default = torch.nn.functional.cross_entropy)
        prob_activation: Callable[[Tensor], Tensor] representing the last activation used to compute the probabilities
                         as output of the classification. The attribute is determined by the multi_label parameter.
                         (default = torch.nn.Softmax(dim=1))
        multi_label: bool specifying if the classification is a multi label one, implying to replace the softmax
                     by a sigmoid at the last layer
                     (default = False)
        f1: F1Measure computing the precision, recall and f1 score during training
            (default : the labels are transformed in a multi_label classification but where only one label is selected)
        i:  int representing the number of times the forward method is called (can be used for showing intermediate
            results)
        debug: bool to indicate if part of the forward call should be show to the user

    Args:
        voc:    AllenNLP Vocabulary object constructed from the tokenizer selected inside the huggingface platform
        labels: Tuple[str, ...] containing the different labels that could be associated with each of the code snippet
                (default = ("success", "failed"))
        huggingface_model: str representing the name of the pretrained model from huggingface you want to use to embed
                           the input tokenized data (should be a feature-extraction model and the same that the
                           tokenizer and indexer you used to read the input data)
                           (default = Enoch/graphcodebert-py)
        kwargs_embedder:   Dict containing the additional args you want to put inside the embedder
                           (default = {})
        embedding_size:    int representing the dimension of the embedding created by the model
                           (default = 768)
        encoder:  Dict corresponding to the keyword arguments used to create the encoder thanks to the create_parameter
                  function
                  (default = create a BertPooler encoder)
        classification_head: Dict corresponding to the keyword arguments used to create the accuracy thanks to the
                             create_parameter function
                             (default = create a simple dense layer (embedding size to num_labels))
        accuracy: Dict corresponding to the keyword arguments used to create the accuracy thanks to the create_parameter
                  function
                  (default = do not use this type of metric (e.g. multi-label architecture))
        loss:  Dict containing the information about the loss that should be used to train the model
               (default = CrossEntropyLoss)
        multi_label: bool specifying if the classification is a multi label one, implying to replace the softmax
                     by a sigmoid at the last layer
                     (default = False)
        debug: bool to indicate if part of the forward call should be show to the user
    """
    __slots__ = ["embedder", "encoder", "classifier", "accuracy", "loss", "prob_activation", "multi_label", "i", "f1",
                 "debug"]

    def __init__(self,
                 voc: Vocabulary,
                 labels: Tuple[str, ...] = ("success", "failed"),
                 huggingface_model: str = "Enoch/graphcodebert-py",
                 kwargs_embedder: Dict[str, Any] = None,
                 embedding_size: int = 768,
                 encoder: Dict[str, Any] = None,
                 classification_head: Dict[str, Any] = None,
                 accuracy: Dict[str, Any] = None,
                 loss: Dict[str, Any] = None,
                 multi_label: bool = False,
                 debug: bool = False
                 ):

        # Labels token supposed in the corresponding namespace for the classification process
        # But we check if it is indeed the case
        labels_voc = voc.get_token_to_index_vocabulary("labels")
        for label in labels:
            if label not in labels_voc:
                voc.add_token_to_namespace(label, "labels")
        # Print the vocabulary for the user
        print(voc)

        # Init the model with the current vocabulary
        super().__init__(voc)

        # ~~~~~~~~~~~~~~~~~~~~

        # Ensure having the default additional parameters to the embedder
        if kwargs_embedder is None:
            kwargs_embedder = {}

        # Creates the embedder for this model
        self.embedder = AutoModel.from_pretrained(huggingface_model,
                                                  **(kwargs_embedder.get("huggingface_parameters", {})))
        if not (kwargs_embedder.get("trainable", False)):
            for param in self.embedder.base_model.parameters():
                param.requires_grad = False

        # ~~~~~~~~~~~~~~~~~~~~

        # If the encoder is not specified, create a BertPooler by default
        if encoder is None:
            encoder = {"name": "bert_pooler", "arg": [huggingface_model], "kwargs": {}}

        # Creates the attribute to encode the code as a single embedding
        self.encoder = create_parameter("encoder", **encoder)

        # ~~~~~~~~~~~~~~~~~~~~

        # Count the labels and creates the linear layer for classifying
        num_labels = len(voc.get_token_to_index_vocabulary("labels"))

        # If the classification head is not specified, create a simple dense layer by default
        if classification_head is None:
            classification_head = {"name": "simple", "arg": [embedding_size, num_labels], "kwargs": {}}

        # Creates the attribute to compute a component for each class
        self.classifier = create_parameter("classification_head", **classification_head)

        # ~~~~~~~~~~~~~~~~~~~~

        # If the accuracy is not specified, we ignore this parameter
        if accuracy is not None:
            # Creates the accuracy metric for the training and evaluation
            self.accuracy = create_parameter("accuracy", **accuracy)
        else:
            self.accuracy = None

        # Creates the metric to follow the precision, recall and f1 score during training (and at the evaluation too)
        self.f1 = F1MultiLabelMeasure(average="micro")

        # ~~~~~~~~~~~~~~~~~~~~

        # Check if the loss is specified, otherwise use the default cross_entropy
        if loss is None:
            loss = {"name": "cross_entropy"}

        # Create the loss function
        self.loss = create_parameter("loss", **loss)

        # ~~~~~~~~~~~~~~~~~~~~

        # Select the type of activation to ue according to the type of classification
        self.prob_activation = torch.nn.Softmax(dim=1) if not multi_label else torch.nn.Sigmoid()

        # Save the multi_label parameter (for computing the scores)
        self.multi_label = multi_label

        # ~~~~~~~~~~~~~~~~~~~~
        
        # Initialize the count of forward calls
        self.i = 0

        # Variable to account for a debugging process
        self.debug = debug

    # -------------------------------

    # Inspired by the official AllenNLP tutorial: https://guide.allennlp.org/training-and-prediction#4
    def forward(self,
                input_ids: torch.Tensor,
                mask: torch.Tensor,
                positions_ids: torch.Tensor,
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Method constructing the network and flowing the data inside it.

        Args:
            input_ids: a Tensor associated with the ids of the tokenized version of the input code.
            mask: a Tensor containing the attention mask to be used inside the embedder.
            positions_ids: a Tensor containing the positional embedding of the input sequence.
            label: a Tensor associated with the input code (None if only predicting without labelled data).

        Returns:
            a Dict of tensors containing the prediction and possibly the loss and accuracy measures
            (if a label is provided).
        """
        # Account for this call
        self.i += 1

        # Gets the generated mask from the tokenizer and the indexer
        mask = mask.bool().to(config["CONFIG"]["device"])

        # Gets the tokens indexed by the reader
        toks = input_ids.long().to(config["CONFIG"]["device"])

        # Gets the positional embeddings to be used
        pos = positions_ids.long().to(config["CONFIG"]["device"])

        # Generates the embedding for the code
        emb = self.embedder(toks, mask, position_ids=pos).last_hidden_state

        # Get the embedding of the total input
        embedded_text = self.encoder(emb)

        # Generates the logits for each batch example
        logits = self.classifier(embedded_text)

        # Probabilities after score output
        probs = self.prob_activation(logits)

        # Debug functionality where the probs and input are shown every 10 calls
        if self.debug and self.i-1 % 10 == 0:
            print(input_ids)
            print(probs)

        # Transform the label input to be usable by the F1 metric (if multi label classification)
        if not self.multi_label and label is not None:
            new_label = torch.zeros((len(label), len(config["MODEL"]["labels"])+2))
            for l in range(len(label)):
                new_label[l][label[l]] = 1
            new_label = new_label.to(config["CONFIG"]["device"])
        else:
            new_label = label
        
        # Puts the probabilities inside the output
        output = {'probs': probs}

        # If a label was provided (training or testing)
        if label is not None:
            # Transform label to correct device
            label = label.to(config["CONFIG"]["device"])
            # Computes the accuracy
            if self.accuracy is not None:
                self.accuracy(logits, label)
            # Computes the f1 score
            self.f1(probs, new_label)
            # Computes the loss associated with the examples (for backpropagation)
            output['loss'] = self.loss(logits, label)

        # Return the generated output for this training step
        return output

    # -------------------------------

    # Inspired by the official AllenNLP tutorial: https://guide.allennlp.org/training-and-prediction#4
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # Method inherited from Model
        # Compute the f1 score and adds it to the metric dictionary
        f1_val = self.f1.get_metric(reset)
        # Compute the accuracy if it is possible to do so
        accuracy_metric = {} if self.accuracy is None else {"accuracy": self.accuracy.get_metric(reset)}
        return {**accuracy_metric, **f1_val}

# ======================================================================================================================


# Inspired by the official AllenNLP tutorial: https://guide.allennlp.org/training-and-prediction#4
def build_code_reader(kwargs_reader: dict[str, Any] = None) -> DatasetReader:
    """
    Function to create the default dataset reader (additional parameters could be added if desired).

    Args:
        kwargs_reader: dict of each parameter we could add to the CodeReader. By default, it creates the
                       default configuration for this object, but overloading the parameters by specifying them inside
                       this dict will prefer them to the original default configuration.

    Returns:
        a CodeReader with the desired parameters.
    """
    if kwargs_reader is None:
        kwargs_reader = {}

    return CodeReader(**kwargs_reader)

# ======================================================================================================================


# Inspired by the official AllenNLP tutorial: https://guide.allennlp.org/training-and-prediction#4
def read_data(reader: DatasetReader, train_path: str, validation_path: str) -> tuple[list[Instance], list[Instance]]:
    """
    Function to recover the examples for the training and validation process of the model.

    Args:
        reader: a DatasetReader able to process the specified files containing the code snippets and the labels.
        train_path: the file path containing all the code examples on which we want to train the model.
        validation_path: the file path on which the model will be evaluated during training.

    Returns:
        the training data and validation data examples as lists of Instance.
    """
    print("-"*50)
    print("Reading data")
    training_data = list(reader.read(train_path))
    validation_data = list(reader.read(validation_path))
    print(f"Stats:\n"
          f"\tNumber of training examples: {len(training_data)}\n"
          f"\tNumber of validation examples: {len(validation_data)}")
    print("-"*50+"\n")
    return training_data, validation_data

# ======================================================================================================================


# Inspired by the official AllenNLP tutorial: https://guide.allennlp.org/training-and-prediction#4
def build_data_loaders(
        train_data: list[Instance],
        validation_data: list[Instance],
        train_batch_size: int = 8,
        validation_batch_size: int = 8
) -> tuple[DataLoader, DataLoader]:
    """
    Function used to create the DataLoader used during the training and validation process.

    Args:
        train_data:       a List of Instance corresponding to the entries found in the training file.
        validation_data:  a List of Instance corresponding to the entries found in the validation file.
        train_batch_size:      an integer for the size of the training batches (default = 8).
        validation_batch_size: an integer for the size of the validation batches (default = 8).

    Returns:
        the constructed DataLoaders.
    """
    # NOTE: Here we use SimpleDataLoader, however, it could be more efficient to use multiprocessing equivalent
    train_loader = SimpleDataLoader(train_data, train_batch_size, shuffle=True)
    validation_loader = SimpleDataLoader(validation_data, validation_batch_size, shuffle=False)
    return train_loader, validation_loader

# ======================================================================================================================


# Inspired by the official AllenNLP tutorial: https://guide.allennlp.org/training-and-prediction#4
def build_trainer(
        model: Model,
        serialization_dir: str,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        num_epochs: int = 1,
        validation_metric: str = "-loss",
        patience: None | int = None,
        learning_rate: float = 1e-5,
        kwargs_optimizer: dict[str, Any] = None
) -> Trainer:
    """
    Function used to build the trainer used to improve the specified model with the training steps.
    By default, the trainer is a GradientDescentTrainer with optimizer set to HuggingfaceAdamWOptimizer for 1 epoch.

    Args:
        model: the AllenNLP Model you want to train.
        serialization_dir: the path of the directory used to store the weights and results.
        train_loader: a DatasetLoader containing the training examples.
        validation_loader:   a DatasetLoader containing the validation examples.
        num_epochs: an int corresponding to the number of epochs you want to perform.
                    (default=1)
        validation_metric: a str representing the criterion used to save the best model.
                           (default="+fscore")
        patience: an integer to specify if early stopping should be activated (not specifying it disables this
                  functionality), this means that, if the model does not improve its performance on the validation
                  dataset for this number of epoch, then the model stops the training and save its best epoch.
                  (default : not activated)
        learning_rate: the learning rate used to learn the weights in the model. This should be a floating point number
                       greater than 0. This is given to HuggingfaceAdamWOptimizer has parameter lr.
                       (default=1e-5)
        kwargs_optimizer: dict containing additional args to be given to the init method of HuggingfaceAdamWOptimizer.
                          (default : no additional parameters)

    Returns:
        the constructed GradientDescentTrainer.
    """
    # Logs for the user
    print(50*"-")
    print("Building the trainer")
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    print(f"Parameters: {len(parameters)}")
    print("\t"+"\n\t".join(f"{i}: {x[0]}" for i, x in enumerate(parameters)))

    # Sets the kwargs to default
    if kwargs_optimizer is None:
        kwargs_optimizer = {}

    # Creates the optimizer
    # NOTE: We could add the optimizer inside the configuration parameters.
    optimizer = HuggingfaceAdamWOptimizer(
        parameters,
        lr=learning_rate,
        **kwargs_optimizer
    )

    # Creates the trainer
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_metric=validation_metric,
        validation_data_loader=validation_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        patience=patience
    )
    print((50 * "-")+"\n")
    return trainer

# ======================================================================================================================


# Inspired by the official AllenNLP tutorial: https://guide.allennlp.org/training-and-prediction#4
def build_model(
        voc: Vocabulary,
        kwargs_model: dict[str, Any] = None
) -> Model:
    """
    Function to create the default model with the parameters set to all the default values.

    Args:
        voc: AllenNLP Vocabulary object constructed from the tokenizer selected inside the huggingface platform.
        kwargs_model: the Dict corresponding to the additional parameters you might want to add to the constructed
                      Model (default = {}).

    Returns:
        the constructed Model.
    """
    print(50*"-")
    print("Building the model")

    if kwargs_model is None:
        kwargs_model = {}
    else:
        print("Additional parameters")
        print(kwargs_model)

    print((50*"-")+"\n")
    return ClassificationEmbedderModel(voc, **kwargs_model)

# ======================================================================================================================


# Inspired by the official AllenNLP tutorial: https://guide.allennlp.org/training-and-prediction#4
def train_model(
        train_path: str,
        validation_path: str,
        voc: Vocabulary,
        serialization_dir: str = None,
        cuda: bool = True,
        epochs: int = 1
) -> tuple[Model, DatasetReader]:
    """
    Function allowing to start the training process of the default model.

    Args:
        train_path:      a str containing the path referencing the training txt file.
        validation_path: a str containing the path referencing the validation txt file.
        voc: a AllenNLP Vocabulary object created upon the pretrained model from huggingface.
        serialization_dir: a str representing the directory in which the results should be stored
                           (default = a tmp file is created to store these results).
        cuda: a boolean indicating if the model should be considered to run on GPU or not.
        epochs: an int representing the number of epochs to perform to train the model.

    Returns:
        the trained Model and the DatasetReader associated with the default parameters after training.
    """
    # Get the dataset reader with the default parameters
    dataset_reader = build_code_reader(config["READER"])

    # Gets the train and validation datasets
    train_data, validation_data = read_data(dataset_reader, train_path, validation_path)

    # Check if the batch normalization was activated
    if config["MODEL"].get("classification_head", {}).get("kwargs", {}).get("norm", False):
        # Get the training batch size
        batch_size = config["CONFIG"]["batch_size"]
        # Get the number of training examples
        train_size = len(train_data)
        # Check if the training set should not be modified
        if train_size % batch_size == 1:
            # The code should be tolerant to misconfiguration
            if train_size == 1:
                raise ValueError("Your training dataset is only composed of a single instance which will lead "
                                 "to error when training the model with normalization as you specified it in the "
                                 "configuration. Ensure this is normal to only have one single training example with "
                                 "such parameters.")
            # Warn the user his/her training dataset has changed
            warnings.warn("The training batch size you are using paired with BatchNormalization will raise an error "
                          f"during training. The training dataset size becomes: {train_size - 1}.")
            train_data = train_data[:-1]

    # Construct the default model
    model = build_model(voc, config["MODEL"])

    # Specifies to be on GPU or not
    if cuda:
        model.to('cuda')
    else:
        model.to('cpu')

    # Converts the data into dataloaders + index with the vocabulary
    train_loader, validation_loader = build_data_loaders(train_data, validation_data, config["CONFIG"]["batch_size"],
                                                         config["CONFIG"]["validation_batch_size"])
    train_loader.index_with(voc)
    validation_loader.index_with(voc)

    # If we should create a temporary file to maintain the results
    if serialization_dir is None:
        # We keep it as in the AllenNLP tutorial to ensure a serialization dir to exist
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = build_trainer(model, tmp_dir, train_loader, validation_loader, epochs, **config["TRAINER"])
            print("Starting training")
            trainer.train()
            print("Finished training\n")
    else:
        trainer = build_trainer(model, serialization_dir, train_loader, validation_loader, epochs, **config["TRAINER"])
        print("Starting training")
        trainer.train()
        print("Finished training\n")

    # Return the trained model and the dataset_reader
    return model, dataset_reader

# ======================================================================================================================


# This class was written by using the AllenNLP official tutorial as inspiration.
# The tutorial can be found at: https://guide.allennlp.org/training-and-prediction#4
# The source code of this tutorial is: https://github.com/allenai/allennlp-guide/blob/master/quick_start/predict.py
# We modified the label handling
class CodeClassifierPredictor(Predictor):
    """
    Predict the labels to be associated with the code snippets
    """

    # Inspired by the official AllenNLP tutorial: https://guide.allennlp.org/training-and-prediction#4
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    # Inspired by the official AllenNLP tutorial: https://guide.allennlp.org/training-and-prediction#4
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        # Method inherited from the super class
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)

    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: dict[str, np.ndarray]
    ) -> list[Instance]:
        # Method inspired by text_classifier predictor in AllenNLP library
        # It transforms an Instance into a new Instance where a label is associated with
        # It allows to add the interpretability components to the system
        new_instance = instance.duplicate()
        if config["MODEL"].get("multi_label", False):
            label = [i for i, v in enumerate(outputs["probs"]) if v > 0.5]
            new_instance.add_field("label", MultiLabelField(label, skip_indexing=True, num_labels=len(outputs["probs"])))
        else:
            label = np.argmax(outputs["probs"])
            new_instance.add_field("label", LabelField(int(label), skip_indexing=True))
        return [new_instance]

    def get_interpretable_text_field_embedder(self) -> torch.nn.Module:
        # Method to get the part of the model which is responsible for the embedding generation
        # Here it is the last layer of the embedder which is the BertPooler
        # If we use another encoder, this would be another component (not implemented)
        # This follows the behaviour of the function we override
        for module in self._model.modules():
            if isinstance(module, transformers.models.roberta.modeling_roberta.RobertaModel):
                return module
        raise ValueError("Did not found a RobertaModel, are you sure to have a model using this layer. "
                         "If this is not the case, please modify / override this method with your current "
                         "architecture.")

# ======================================================================================================================


class CodeClassifierInterpreter:
    """
    Class allowing to construct a Saliency Interpreter which gives the tokens that should be the one the more
    probable to be the reason of such label associated with the current code.

    Attributes:
        predictor: a Predictor object we can use for labelling new entries'
        dataset_reader: a CodeReader object to transform new entries' to Instances
        interpreter: the Saliency Map interpreter used

    Args:
        predictor: a Predictor object we can use for labelling new entries'
        dataset_reader: a CodeReader object to transform new entries' to Instances
    """
    __slots__ = ["predictor", "dataset_reader", "interpreter"]

    def __init__(self, predictor: Predictor, dataset_reader: CodeReader):
        self.predictor = predictor
        self.dataset_reader = dataset_reader
        self.interpreter = IntegratedGradient(predictor)

    def interpret_json(self, inputs: dict[str, str]) -> dict[str, dict[str, Sequence[float]]]:
        """
        Get an inputs formatted as a json and send it to the Saliency Map interpreter

        Args:
            inputs: an entry formatted as a json

        Returns:
            the score associated with each entry
        """
        return self.interpreter.saliency_interpret_from_json(inputs)

    def interpret(self, sentence: str, limit=5) -> list[tuple[int, tuple[str, float]]]:
        features: dict[str, str] = dataset_reader.get_features(sentence)

        inter = self.interpret_json({"sentence": sentence})
        inter = inter["instance_1"]["grad_input_1"]

        f = [x.replace('\u0120', '_') for x in features["input_tokens"]]
        inter_ids = list(zip(f, inter))
        inter_ids_pos = list(enumerate(inter_ids))

        inter_sort = sorted(inter_ids_pos, key=lambda x: x[1][1], reverse=True)[:limit]

        return inter_sort

# ======================================================================================================================


async def handle_msg(
        msg: bytes,
        reader: StreamReader,
        writer: StreamWriter,
        predictor: CodeClassifierPredictor,
        interpreter: CaptumInterpreter
) -> str:
    """
    Asynchronous function in charge of answering to a specific message according to the prediction the model gives
    for this newly received input

    Args:
        msg:    bytes message received over the reader and corresponding to the code for which a prediction has to be
                made
        reader: StreamReader over which the message has been received
        writer: StreamWriter over which the prediction will be sent
        predictor: CodeClassifierPredictor used to associate probabilities to each received message
        interpreter: CaptumInterpreter to compute the scores of each token in a prediction

    Returns:
        the decoded message received
    """
    # Any error is ignored to avoid crashing
    try:
        # Get the input code the user sent to the model
        input_code: str = msg.decode("latin1")
        input_code = input_code.strip()
        print(f"Received {input_code}")

        # If the input code is the "connection_ending" one, inform the model it can close the connection
        if input_code == "$$$":
            return "QUIT"

        # Check if the code should be interpreted for a particular label
        if input_code[:3] == "%%%":
            # Get the label and the code of the request
            interpret_label, input_code = input_code[3:].split("%x%")
            interpret_index = config["MODEL"]["labels"].index(interpret_label) + 2

            # Ask the interpreter
            out_interpret = interpreter.interpret(input_code, (interpret_index, interpret_label, 0.0), quiet=True,
                                                  device=config["CONFIG"]["device"])

            # Manually compute the offset of each token (since the offset from the tokenizer does not match the
            # tokenization procedure followed by Microsoft)
            offsets = [(0, 0)]
            end_last_token = 0
            # Get the tokenizer and use the same pipeline used to tokenize the code entry
            tokenizer = dataset_reader.tokenizer.tokenizer
            code_tokens = get_code_tokens(input_code, tokenizer, dataset_reader.compiled_language)
            # Go over each token
            for token in code_tokens:
                # We do not consider the "_" char as a useful char for the offset
                if token[0] == "\u0120":
                    token = token[1:]
                # However, since tabs and line feeds are not part of the tokenized code, the start of this token
                # is shifted until we find a match between the code and the token
                while end_last_token < len(input_code) and \
                        token != input_code[end_last_token:end_last_token+len(token)]:
                    end_last_token += 1
                # The offset if the range (start_i, end_i) such that token_i = code[start_i:end_i]
                offsets.append((end_last_token, end_last_token+len(token)))
                # The next token starts at the end of this token
                end_last_token += len(token)

            # Go over the tokens to construct the reply
            reply = []
            # Get the most important token from the interpreter
            for idx, tok_info in out_interpret:
                # Check that this token has an offset
                if 0 <= idx < len(offsets):
                    # Each token range is constructed as "start,end:score"
                    reply.append(f"{offsets[idx][0]},{offsets[idx][1]}:{tok_info[1]}")
            # Each token is separated by a ";"
            to_send = ";".join(reply)

        # Otherwise, it's a simple prediction request
        else:
            # Get the probabilities predicted by the model (the two first labels (default ones) are never predicted)
            probs: list[float] = predictor.predict(input_code)["probs"][2:]

            # Transform the prediction in a list of "LABEL:VALUE"
            values = []
            for l, p in zip(config["MODEL"]["labels"], probs):
                values.append(l+":"+str(p))
            to_send = ";".join(values)  # Each pair is separated by a ";"

        # Encode the predictions to be sent over the writer
        probs_bytes = b""
        probs_bytes += bytes(to_send, "latin1")
        probs_bytes += b"\r\n"

        # Send this new message by writing over the writer
        writer.write(probs_bytes)

        # Inform the user of the server what was sent over the connection
        print(f"Sent {to_send}")

        # Return the original input message
        return input_code

    except Exception as e:
        print(e)
        return ""


async def discuss(reader: StreamReader, writer: StreamWriter) -> None:
    """
    Asynchronous function allowing to serve continuously a client that connected itself to the model

    Args:
        reader: StreamReader representing the incoming connection from the connected client
        writer: StreamWriter representing the outgoing connection to the connected client
    """
    print("Accepting a client")

    # Creating a new predictor to serve this client
    predictor = CodeClassifierPredictor(model, dataset_reader)

    # Create a new interpreter to maybe serve the client if it requests it
    interpreter = CaptumInterpreter(model, dataset_reader, predictor,
                                    **config["INTERPRETER"].get("kwargs", {}))

    msg = ""  # The last received message

    # While the connection was not abruptly closed and not closed explicitly by the client
    while reader is not None and msg != "QUIT":
        # The different inputs should be separated by "$x$"
        msg = await reader.readuntil(b"$x$")
        # The message is handled and the response sent
        msg = await handle_msg(msg[:-3], reader, writer, predictor, interpreter)
    return


async def serve() -> None:
    """
    Asynchronous function responsible for continuously listening to incoming connection on the local host, port 8000
    """
    print("Started serving")
    # NOTE: We should add the IP address inside the yaml configuration instead of using "" (aka localhost) each time
    srv: asyncio.AbstractServer = await asyncio.start_server(discuss, "", 8000)
    # Each time a client connects, the discuss function is called asynchronously
    # (allow serving multiple clients at the same time)
    await srv.serve_forever()

# ======================================================================================================================


if __name__ == "__main__":
    import argparse

    # Parser for command line execution
    parser = argparse.ArgumentParser(description="Program manipulating the AllenNLP library to train, evaluate and "
                                                 "predict from a code snippet classifier. The base embedder used in "
                                                 "this architecture is GraphCodeBERT from Microsoft. The system will first "
                                                 "tokenize the input data where each snippet is composed by first the "
                                                 "plain code followed by the associated label (by default failed or "
                                                 "success) separated by the code ' $x$ '. The snippets are separated "
                                                 "by the code '\\n$$$\\n'.\n"
                                                 "Part of this code have been written by following the AllenNLP "
                                                 "official tutorial (https://guide.allennlp.org/training-and-prediction#4).\n"
                                                 "Author: Guillaume Steveny ; Year: 2023--2024")

    parser.add_argument("-p", "--predict", action="store_true",
                        help="If the interactive tool for prediction should be started after the evaluation.")

    parser.add_argument("-n", "--no_loop", action="store_true",
                        help="If the training should be omitted.")

    parser.add_argument("-ne", "--no_eval", action="store_true",
                        help="If the evaluation should be omitted")

    parser.add_argument("-g", "--gui", action="store_true",
                        help="If the tool should be started linked with a gui tool "
                             "(should have no_loop and no_eval set)")

    parser.add_argument("-m", "--model", action="store", type=str, default="Enoch/graphcodebert-py",
                        help="The name of the feature extraction model that should be used "
                             "(default=Enoch/graphcodebert-py).")

    parser.add_argument("-lm", "--load_model", action="store", type=str, default=None,
                        help="The path to the .th file where the weights of the trained model are stored. "
                             "This parameter is only used when associated with the 'no_loop' parameter. "
                             "If you do not specify a value for this 'load_model' parameter, the model will try "
                             "to be recovered from 'serialization_dir'/'best.th'. Otherwise, your path value will be "
                             "used.")

    parser.add_argument("-t", "--training", action="store", type=str, default="../output/codebert/train.txt",
                        help="The path to the training txt file containing the code snippets to train on "
                             "(default=../output/codebert/train.txt).")

    parser.add_argument("-v", "--validation", action="store", type=str, default="../output/codebert/validation.txt",
                        help="The path to the validation txt file containing the code snippets for validation "
                             "(default=../output/codebert/validation.txt).")

    parser.add_argument("-e", "--evaluation", action="store", type=str, default="../output/codebert/test.txt",
                        help="The path to the test txt file containing the code snippets for evaluation "
                             "(default=../output/codebert/test.txt).")

    parser.add_argument("-r", "--random_seed", action="store", type=int, nargs="?", default=None, const=42,
                        help="Specifies if the model should be trained with a specific seed for the random number "
                             "generator contained in pytorch, numpy and the python random module for results "
                             "reproducibility. Not specifying it does not set a seed, just giving the parameter "
                             "without value sets the seeds to 42. Otherwise it uses the value you specified.")

    parser.add_argument("-s", "--serialization_dir", action="store", type=str, default="../output/cesres_codebert",
                        help="The path to the directory in which to store the metrics at each epoch, the best model "
                             "and the final results and predictions"
                             "(default=../output/cesres_codebert).")

    parser.add_argument("-b", "--batch_size", action="store", type=int, default=8,
                        help="The size of batches to use when training the model. "
                             "(default=8)")

    parser.add_argument("-vb", "--validation_batch_size", action="store", type=int, default=None,
                        help="The size of batches to use when performing the validation step during training. "
                             "(default : same value as 'batch_size' parameter)")

    parser.add_argument("-eb", "--evaluation_batch_size", action="store", type=int, default=None,
                        help="The size of batches to use when performing the evaluation step after training. "
                             "(default : same value as 'batch_size' parameter)")

    parser.add_argument("-l", "--loops", action="store", type=int, default=1,
                        help="The number of epochs to train the model."
                             "(default=1)")

    parser.add_argument("-c", "--config", action="store", type=str, default=None,
                        help="Path to a configuration yaml file specifying all the parameters the model could use. "
                             "Omitting this option or parameters in this file will take the default parameters.")

    parser.add_argument("-d", "--device", action="store", type=str, default="cpu",
                        help="Name of the PyTorch device on which the model should be loaded and used when performing "
                             "the evaluation and prediction "
                             "(default=cpu).")

    # -------
    # Gets the args
    args = parser.parse_args()

    # Retrieve the parameters inside the yaml file if provided
    if args.config:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
        for p in ["VOCABULARY", "READER", "MODEL", "TRAINER", "INTERPRETER", "CONFIG"]:
            if p not in config:
                config[p] = {}
        for k in vars(args):
            if k not in config["CONFIG"]:
                config["CONFIG"][k] = vars(args)[k]
    else:
        config: dict[str, dict[str, Any]] = {
            "VOCABULARY": {},
            "READER": {},
            "MODEL": {},
            "TRAINER": {},
            "INTERPRETER": {},
            "CONFIG": vars(args)
        }

    # -------
    # Check if the model runs in GUI mode, the other parameters (training and predicting) should be disabled
    if config["CONFIG"]["gui"]:
        if not config["CONFIG"]["no_loop"] or not config["CONFIG"]["no_eval"]:
            raise ValueError("Parameter 'gui' should be the same as 'no_loop' and 'no_eval'")
        if config["CONFIG"]["predict"]:
            raise ValueError("Parameter 'gui' should not be set if 'predict' is on")

    # -------
    # Check the validation for the validation and evaluation batch size
    if config["CONFIG"]["validation_batch_size"] is None:
        config["CONFIG"]["validation_batch_size"] = config["CONFIG"]["batch_size"]
    if config["CONFIG"]["evaluation_batch_size"] is None:
        config["CONFIG"]["evaluation_batch_size"] = config["CONFIG"]["batch_size"]

    # -------
    # The no_loop_weigh_file parameter should only be used when no_loop
    if not config["CONFIG"]["no_loop"] and "no_loop_weight_file" in config["CONFIG"]:
        # Warn the user of the modification
        warnings.warn("The configuration parameter 'no_loop_weight_file' should only be provided when setting "
                      "no_loop to True. This parameter will be set to its default value (best.th).")
        # Put the default value
        config["CONFIG"]["no_loop_weight_file"] = "best.th"

    # -------
    # Put the right name for the model_path (only if the no_loop parameter is set, otherwise use the default value)
    if config["CONFIG"]["load_model"] is None or not config["CONFIG"]["no_loop"]:

        # Warn the user that the path it used will be ignored
        if config["CONFIG"]["load_model"] is not None:
            warnings.warn(
                "You provided a load_model path in your configuration but specified you wanted the model to "
                "be trained. But the weight file created by the training will be stored in "
                "the serialization dir under the name 'best.th'. The path you provided will then be "
                "ignored.")

        # Put the default name
        config["CONFIG"]["load_model"] = config["CONFIG"]["serialization_dir"] + "/" + \
                                         config["CONFIG"].get("no_loop_weight_file", "best.th")

    # -------
    # Check the compiled language and use default for Windows
    if "compiled_language" not in config["READER"]:
        if os.name == 'nt':
            config["READER"]["compiled_language"] = "./my-language"

    # -------
    # Show the configuration on screen
    print(json.dumps(config, indent=4))
    print()

    # -------
    # Sets the seeds if asked
    if config["CONFIG"]["random_seed"] is not None:
        transformers.set_seed(config["CONFIG"]["random_seed"])

    # -------
    # Get the model for the vocabulary
    huggingface_model = config["VOCABULARY"].get("huggingface_model", args.model)

    # Creates the vocabulary from the specified model
    voc = Vocabulary.from_pretrained_transformer(huggingface_model)

    # Add the labels to the vocabulary
    for label in config["MODEL"].get("labels", ["success", "failed"]):
        voc.add_token_to_namespace(label, "labels")

    # -------
    # Load the model from stored file
    if config["CONFIG"]["no_loop"]:
        #testing pytorch version for loading incompatibility (Jorge)
        print("torch version: ", torch.__version__)
        # Load the weights
        print("load model address: ", config["CONFIG"]["load_model"])
        print("torch device: ", torch.device(config["CONFIG"]["device"]))
        m = torch.load(config["CONFIG"]["load_model"], map_location=torch.device(config["CONFIG"]["device"]))
        # Create the model
        model = ClassificationEmbedderModel(voc, **config["MODEL"])
        # Map the weights to the model
        model.load_state_dict(m)
        # Build the dataset reader
        dataset_reader = build_code_reader(config["READER"])

    # The model and the datasetReader are recovered after training
    else:
        # NOTE: add these lines to track the CO2 emissions
        # tracker = OfflineEmissionsTracker(gpu_ids=[0],
        #                                   output_dir=config["CONFIG"]["serialization_dir"] + "/../energy/",
        #                                   country_iso_code="BEL")
        # tracker.start()
        model, dataset_reader = train_model(config["CONFIG"]["training"], config["CONFIG"]["validation"], voc,
                                            config["CONFIG"]["serialization_dir"],
                                            cuda=config["CONFIG"]["device"] == "cuda", epochs=config["CONFIG"]["loops"])
        # emissions: float = tracker.stop()
        # print(50 * "-" + f"\n\tEmissions: {emissions} kg\n" + "-" * 50)

    # We transform back the model to cpu for the test evaluation (no more training)
    model.to(config["CONFIG"]["device"])

    # -------
    # Gets the test data
    if not config["CONFIG"]["no_eval"]:
        test_data = list(dataset_reader.read(config["CONFIG"]["evaluation"]))

        # Gets the data loader for this test data
        data_loader = SimpleDataLoader(test_data, batch_size=config["CONFIG"]["evaluation_batch_size"],
                                       shuffle=False)
        # Index the data
        data_loader.index_with(voc)

        # Gets the results of the evaluation of the best model
        m = torch.load(config["CONFIG"]["load_model"], map_location=torch.device(config["CONFIG"]["device"]))
        model = ClassificationEmbedderModel(voc, **config["MODEL"])
        model.load_state_dict(m)
        # The loaded model is converted to the current device
        model.to(config["CONFIG"]["device"])
        # Call 'evaluate' from allennlp which perform the complete output generation
        results = evaluate(model, data_loader, cuda_device=0 if config["CONFIG"]["device"] == "cuda" else -1,
                           output_file=config["CONFIG"]["serialization_dir"]+"/test_results.txt",
                           predictions_output_file=config["CONFIG"]["serialization_dir"]+"/predictions.txt")
        # Print the results dict to the user
        print(results)

    # -------
    # If the interactive prediction tool is activated
    if config["CONFIG"]["predict"]:
        # Show user welcome and help
        print("This command line tool allows you to predict code snippet from your provided input.")
        print("You can write your code snippets one line after the other. "
              "Ending an input is done by double typing 'enter'")
        print("Typing '$$$' as a line exits the tool at whatever writing step.\n")
        # Sets the model in eval mode
        model.eval()
        # Create the predictor and interpreter
        predictor = CodeClassifierPredictor(model, dataset_reader)
        # Get if the captum mode is enabled
        captum_mode = config["INTERPRETER"].get("captum", False)
        if captum_mode:
            interpreter = CaptumInterpreter(model, dataset_reader, predictor,
                                            **config["INTERPRETER"].get("kwargs", {}))
        else:
            interpreter = CodeClassifierInterpreter(predictor, dataset_reader)
        # Variables used to predict
        exit_var = False
        code = ""
        last = ""
        # While the user did not used $$$ to quit
        while not exit_var:
            # Wait for a line from the user
            val = input("> ")
            # Using $$$ stops the predictor
            if val == "$$$":
                exit_var = True
            else:
                # If the user pressed twice the enter button
                if val == "" and last == "":
                    # Print back the complete code to the user
                    print("\n"+code)
                    # Get all the actual predictions
                    prediction = predictor.predict(code)["probs"][2:]
                    # For each, print the label associated
                    print("Predicted labels")
                    for l, p in zip(config["MODEL"]["labels"], prediction):
                        print(f"\t{l} -> {p}")
                    # Perform the interpretation part
                    print("\nInterpretation part")
                    if captum_mode:
                        for i, lp in enumerate(zip(config["MODEL"]["labels"], prediction)):
                            if lp[1] > 0.5:
                                interpret = interpreter.interpret(code, (i+2, lp[0], lp[1]),
                                                                  device=config["CONFIG"]["device"])
                                print()
                    else:
                        interpret = interpreter.interpret(code)
                        for t in interpret:
                            print(f"\tToken n°{t[0]}: {t[1][0]} -> {t[1][1]}")
                    # Go to the next input of the user
                    code = ""
                    last = ""
                else:
                    code += val + "\n"
                    last = val

    # -------
    # Launch the model in server mode to serve the CodeLabellingGUI program
    if config["CONFIG"]["gui"]:
        # Sets the model in eval mode
        model.eval()
        print("Starting the server")
        asyncio.run(serve())
