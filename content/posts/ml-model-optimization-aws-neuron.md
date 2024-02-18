---
title: "ML model optimization using AWS Neuron SDK"
date: 2024-02-18T10:22:06
tags: ["ml model", "ml model optimization", "aws neuron", "inferentia"]
---

I had a chance to work with AWS Neuron SDK to optimize ML models to enable running inference endpoints Inferentia based instances. In this post, I give some insights about Neuron and how to optimize models using Neuron SDK.

A quick intro about Neuron SDK, AWS Neuron is the SDK used to run deep learning workloads on AWS Inferentia and AWS Trainium based instances. It supports customers in their end-to-end ML development lifecycle to build new models, train and optimize these models, and then deploy them for production. [1](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/)

AWS Neuron is tailored to leverage the full potential of AWS Inferentia chips, which are custom hardware accelerators for machine learning inference. By using Neuron, you can achieve high-throughput and low-latency inference performance, which is crucial for real-time applications.

It support popular machine learning frameworks such as TensorFlow, PyTorch, and MXNet. It's easy to deploy models developed in these frameworks onto Inferentia chips without significant changes to your existing codebase.

Running inference endpoints on AWS Inferentia chips can be more cost-effective compared to using general-purpose GPUs or CPUs, especially at scale. The Neuron SDK helps in optimizing your models to run efficiently on Inferentia, potentially lowering your cloud infrastructure costs.

To have the performance of Inferentia and Trainium based instances, trained models needs to be optimized with Neuron SDK. For now Neuron SDK is only supported on Linux x86_64 and the best way to use Neuron for optimization is running it on a Deep Learning AMI Neuron PyTorch 1.13 (Ubuntu 20.04) based image. The linux packages, Neuron SDK and other required packages are already available in this image. Let's create an EC2 image and optimize a pre-trained Huggingface model.

Here we have a small Python module that uses `bert-base-cased-finetuned-mrpc` from [transformers](https://pypi.org/project/transformers/) library:

```python
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
```

Upload this file to EC2 instance after creating the instance is created and started running. Once we connect to instance via SSH we see a screen like below

![ec2-welcome-screen](/images/post_pics/ml-model-optimization-aws-neuron/ec2.png)

We need to activate the Python virtual environment that has the Neuron modules installed

```sh
source /opt/aws_neuron_venv_pytorch_inf1/bin/activate
```

Then we can run our optimization module, it downloads the pre-trained tokenizer and model then starts optimizing the model. The output model artifact should be ready in same directory.

```shell
python neuron_optimization.py
```

I hope you find this post interesting and useful :)
