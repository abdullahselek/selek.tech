---
title: "Static vs Dynamic Quantization in Machine Learning"
date: 2024-06-01T18:51:59+01:00
tags: ["deep learning", "machine learning", "ml model", "model", "quantization", "model compression"]
---

It has been a long time since I've shared any post so I thought today is the time for that :) In this post I'll walk you through the details between static vs dynamic quantization which I think might be interesting. Let's start!

What is quantization? Quantization is a process used in machine learning to reduce the precision of the numbers representing the model parameters, which can lead to smaller model sizes and faster inference times. There are different types of quantization, primarily static and dynamic quantization, each with its own set of advantages and use cases. Here I have [another post](https://selek.tech/posts/deep-learning-model-quantization/) which also has some other details about quantization.

## Static Quantization

Static quantization converts the weights and activates of a neural network to lower precision (e.g., from 32-bit floating-point to 8-bit integers) during the training or post-training phase. Here we have a more detailed breakdown of static quantization:

1. Calibration Phase

- A calibration step is performed where the model runs on a representative dataset. This step is important as it helps to gather the distribution statistics of the activations, which are then used to determine the optimal scaling factors (quantization parameters) for each layer.

2. Quantization Parameters

- In this step, the model weights are quantized to a lower precision format (e.g., int8). The scale and zero-point for each layer are computed based on the calibration data and are fixed during inference.

3. Inference

- During inference, both the weights and activations are quantized to int8. Since the quantization parameters are fixed, the model uses these pre-determined scales and zero-points to perform fast, integer-only computations.

4. Performance

- Static quantization typically results in more efficient execution compared to dynamic quantization because all the computations can be done using integer arithmetic, which is faster on many hardware platforms. It often achieves better accuracy compared to dynamic quantization since the quantization parameters are finely tuned using the calibration data.

### Use Cases of Static Quantization

Static quantization is well-suited for scenarios where the input data distribution is known and can be captured accurately during the calibration phase. It's commonly used in deploying models on edge devices where computational resources are limited.

Here's a code sample demonstrating static quantization using PyTorch:

```python
import torch
import torchvision
import torch.quantization as quant

# Load a pre-trained model
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# Define the quantization configuration
model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

# Prepare the model for static quantization
model_prepared = torch.quantization.prepare(model, inplace=False)

# Calibrate the model with representative data
# Here we just run a few samples through the model
for _ in range(10):
    input_tensor = torch.randn(1, 3, 224, 224)
    model_prepared(input_tensor)

# Convert the model to quantized version
model_quantized = torch.quantization.convert(model_prepared, inplace=False)

# Save the quantized model
torch.save(model_quantized.state_dict(), "quantized_model.pth")
```

## Dynamic Quantization

Dynamic quantization quantizes only the weights to a lower precision and leaves the activations in floating-point during the model's runtime. Deeper look at dynamic quantization:

1. No Calibration Needed

- Dynamic quantization does not require a separate calibration phase. The quantization parameters are determined on-the-fly during inference. This makes it more straightforward to apply since it eliminates the need for a representative calibration dataset.

2. Quantization Parameters

- Model weights are quantized to lower precision as int8 format before inference. During inference, activations are dynamically quantized, which means their scale and zero-point are computed for each batch or layer during execution.

3. Inference

- Weights are stored and computed in int8, but activations remain in floating-point until they are used in computations. This allows the model to adapt to the variability in input data at runtime by recalculating the quantization parameters dynamically.

4. Performance

- Dynamic quantization typically incurs a lower reduction in model accuracy compared to static quantization since it can adapt to changes in input data distribution on-the-fly. However, it may not achieve the same level of inference speedup as static quantization because part of the computation still involves floating-point operations.

5. Use Cases:

- Dynamic quantization is particularly useful in scenarios where the input data distribution may vary and cannot be easily captured by a single representative dataset. It is often used in server-side deployments where computational resources are less constrained compared to edge devices.

Sample code of dynamic quantization with PyTorch

```python
import torch
import torchvision

# Load a pre-trained model
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# Apply dynamic quantization
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save the quantized model
torch.save(model_quantized.state_dict(), 'dynamic_quantized_model.pth')
```

## Static Quantization Workflow

1. Model Training: Train your model normally.
2. Calibration: Run the model on a representative dataset to determine quantization parameters.
3. Quantization: Convert model weights and activations to lower precision using fixed quantization parameters.
4. Inference: Perform fast, integer-only inference.

## Dynamic Quantization Workflow

1. Model Training: Train your model normally.
2. Quantization: Convert model weights to lower precision.
3. Inference: Dynamically quantize activations during inference, allowing for adaptable performance based on input data.

## Summary

Both static and dynamic quantization offer ways to reduce the model size and improve inference efficiency but cater to different use cases and trade-offs:

- Static Quantization requires a calibration step, uses fixed quantization parameters, offers faster inference with purely integer arithmetic, and is ideal for scenarios with known and stable input data distributions.
- Dynamic Quantization skips the calibration step, uses dynamically computed quantization parameters during inference, offers more flexibility with input data variability, and provides a simpler application process at the cost of slightly less inference efficiency compared to static quantization.

Choosing between static and dynamic quantization depends on the specific requirements of the deployment environment, such as the stability of the input data distribution, the available computational resources, and the acceptable trade-offs between inference speed and model accuracy.
