import os

import torch
import torch.neuron
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Setting up NeuronCore groups for inf1.6xlarge with 16 cores
num_cores = 4  # This value should be 4 on inf1.xlarge and inf1.2xlarge
os.environ["NEURON_RT_NUM_CORES"] = str(num_cores)

# Build tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased-finetuned-mrpc", return_dict=False
)

# Setup some example inputs
sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

max_length = 128
paraphrase = tokenizer.encode_plus(
    sequence_0,
    sequence_2,
    max_length=max_length,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)
not_paraphrase = tokenizer.encode_plus(
    sequence_0,
    sequence_1,
    max_length=max_length,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)

# Run the original PyTorch model on compilation exaple
paraphrase_classification_logits = model(**paraphrase)[0]

# Convert example inputs to a format that is compatible with TorchScript tracing
example_inputs_paraphrase = (
    paraphrase["input_ids"],
    paraphrase["attention_mask"],
    paraphrase["token_type_ids"],
)
example_inputs_not_paraphrase = (
    not_paraphrase["input_ids"],
    not_paraphrase["attention_mask"],
    not_paraphrase["token_type_ids"],
)

# Run torch.neuron.trace to generate a TorchScript that is optimized by AWS Neuron
model_neuron = torch.neuron.trace(model, example_inputs_paraphrase)

# Verify the TorchScript works on both example inputs
paraphrase_classification_logits_neuron = model_neuron(*example_inputs_paraphrase)
not_paraphrase_classification_logits_neuron = model_neuron(
    *example_inputs_not_paraphrase
)

# Save the TorchScript for later use
model_neuron.save("bert_neuron.pt")
