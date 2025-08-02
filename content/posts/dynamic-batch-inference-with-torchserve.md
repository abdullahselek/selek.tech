---
title: "Dynamic Batch ML Inference with TorchServe"
date: 2023-12-25T16:30:45
tags: ["dynamic batch inference", "ml model serving", "ml", "torchserve"]
---

Recently I've done research on ML model serving frameworks and worked on dynamic batch inference applications. I thought it would be great to share some details here which you might find useful. A quick intro about ML model serving first.

## Machine Learning (ML) Model Serving

Machine Learning model serving refers to the process of deploying a trained ML model into a production environment, where it can be used to make predictions or inferences based on new, real time data. This process is a critical phase in the life cycle of an ML model, as it transitions from a development phase, where it's trained and tested, to being an integral part of an application or service.

There are many options as model serving frameworks such as [BentoML](https://github.com/bentoml), [Cortex](https://github.com/cortexlabs/cortex), [Tensorflow Serving](https://github.com/tensorflow/serving) but in this post we are going to focus on [TorchServe](https://pytorch.org/serve/index.html) which also support dynamic batching.

## Dyanmic Batch Inference

Dynamic batch inference is a technique used in machine learning model serving where inference requests are grouped into batches dynamically to optimize processing efficiency and resource utilization. This approach is particularly relevant when dealing with models that are computationally expensive and when serving requests at a large scale.

Let's check the key aspects of dynamic batch inference:

1. Batch Processing, in machine learning, running inferences on a batch of data, rather than individual data points, can significantly improve processing speed. This is because batch processing allows for more efficient use of computational resources, such as GPUs or TPUs, by parallelizing operations across multiple data points.

2. Dynamic Batching, unlike static batching where the batch size is fixed, dynamic batching adjusts the batch size in real time based on the incoming request load. When requests arrive, instead of processing each one immediately, the system temporarily holds them to form a batch. The size of this batch can vary depending on the current load and the configured maximum wait time.

3. Latency vs. Throughput Trade off. Dynamic batch inference often involves a trade off between latency and throughput. Larger batches can lead to higher throughput (more requests processed per unit of time) but might increase latency for individual requests, as they have to wait for other requests to form a full batch. The system must balance this to meet the desired service level agreements.

4. Resource Utilization, by processing requests in batches dynamic batching makes more efficient use of computational resources. This is especially important in environments where resources are limited or costly.

5. Applicability, dynamic batch inference is particularly beneficial in scenarios where the model serving infrastructure experiences variable request loads. For instance, during peak times, the system can form larger batches to handle the high request volume more efficiently.

6. Implementing dynamic batching can be challenging, as it requires a sophisticated queuing system to manage incoming requests and form optimal batches. Additionally, not all models may benefit equally from batch processing, especially if they are not designed to handle batched inputs efficiently.

Now let's make a quick sample implementation using Huggingface's `transformers`, `torch` and `torchserve`.

## Sample Dynamic Batch Serving

First thing is installing the required Python modules.

```shell
pip install torch transformers torchserve torch-model-archiver
```

TorchServe requires JDK installed, I recommend to use Amazon Corretto which you can find the installation details [here](https://aws.amazon.com/corretto).

In this sample we are going to make text classification using `bert-base-uncased` model available in Huggingface. Let's download and save our model and tokenizer. Give this Python file a name and then run, the model and the config files will be available under `model` folder.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Load pre-trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Save the model and tokenizer
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')
torch.save(model.state_dict(), './model/pytorch_model.bin')
```

Next step is to create a `TorchServe` handler for dynamic batch inference. The handler inherits from `BaseHandler` and overrides functions. Create a new Python file and name it `handler.py`. It initializes by loading the model, tokenizer and creates the pipeline using TorchServe's configuration. When a request comes it preprocess the data, apply postprocess and then return the final result.

```python
import os
from typing import Any, List

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from ts.context import Context
from ts.torch_handler.base_handler import BaseHandler


class TransformersClassifierHandler(BaseHandler):
    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()
        self.initialized = False

    def load_pipeline(self, context: Context) -> TextClassificationPipeline:
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        config_path = os.path.join(model_dir, "config.json")

        model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config_path)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, config=config_path)
        return TextClassificationPipeline(model=model, tokenizer=tokenizer, device="cpu", return_all_scores=True)

    def initialize(self, context: Context):
        self.initialized = True
        self.model_pipeline = self.load_pipeline(context=context)

    def preprocess(self, data: List[dict]) -> List[dict]:
        preprocessed_data = data[0].get("data")
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")
        return preprocessed_data            

    def inference(self, input: List[dict]) -> List[List[Any]]:
        classifications = []
        for data in input:
            query = data.get("query")
            if query:
                classification = self.model_pipeline(query)
                classifications.append(classification)
            else:
                classifications.append([])
        return classifications

    def postprocess(self, output: List[List[Any]]) -> List[List[List[Any]]]:
        return [output]

    def handle(self, data: List[dict], context: Context) -> List[List[List[Any]]]:
        model_input = self.preprocess(data=data)
        model_output = self.inference(input=model_input)
        return self.postprocess(output=model_output)
```

Next step is to create the TorchServe model using torch-model-archiver module, the command below generates a arhcieve file under `model_store` folder.

```shell
torch-model-archiver --model-name "bert-text-classifier" \
                     --version 1.0 \
                     --model-file ./model/pytorch_model.bin \
                     --serialized-file ./model/pytorch_model.bin \
                     --handler ./handler.py \
                     --extra-files "./model/config.json,./model/vocab.txt" \
                     --export-path model_store \
                     --force
```

Optional final step is to create a `config.properties` file for TorchServe, for simplicity we will add only two items:

```shell
batch_size=8
max_batch_delay=1000
```

Now we are good to start our dynamic batching inference server by command

```shell
torchserve --start --ncs --model-store model_store --models bert-text-classifier.mar
```

## Testing Dynamic Batch Serving

- We can check the health of the system from `ping` endpoint available at [http://localhost:8080/ping](http://localhost:8080/ping). `curl http://localhost:8080/ping` should work well.
- As we can serve multiple models we can also check the list of served models from [http://localhost:8081/models/](http://localhost:8081/models/). For our sample project `curl http://localhost:8081/models/` returns

```shell
{
  "models": [
    {
      "modelName": "bert-text-classifier",
      "modelUrl": "bert-text-classifier.mar"
    }
  ]
}
```

- To test our prediction endpoint should be ready at [http://localhost:8080/predictions/bert-text-classifier](http://localhost:8080/predictions/bert-text-classifier). We can test it by using POST requests.

```shell
curl --header "Content-Type: application/json" \
--request POST \
--data '[{"query": "TorchServe dynamic batching framework"}, {"query": "Tensorflow Serving"}]' \
http://localhost:8080/predictions/bert-text-classifier
```

That's all I want to share for now about dynamic batching inference. TorchServe provides a powerful framework and infrastructure to facilitate the efficient deployment and management of ML models in production. Hope you enjoyed reading this post and even tried using it.
