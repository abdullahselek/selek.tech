---
title: "DeepSeek R1 Model"
date: 2025-01-27T22:07:20Z
tags:
  - deepseek
  - r1 model
  - mixture of experts
  - reaning model
image:
comments: true
---

Last week Chinese AI labs DeepSeek released their latest reasoning model R1, their models are on par with the most advanced models from OpenAI, Anthropic and Meta. This post is about the details of DeepSeek R1 model and the architecture behind it. Their paper is available at their repo on [Github](https://github.com/deepseek-ai/DeepSeek-R1).

The DeepSeek R1 leverages a Mixture of Experts (MoE) architecture. It is a design where multiple specialized neural networks "experts" collaborate, with only a subset activated for each input. This balances computational efficiency with high performance. Each expert is a feed-forward neural network (FFN), a standard neural network layer that processes inputs through weighted connections and non-linear activations. R1 employs hundreds of these experts, each implicitly specializing in domains like mathematics, coding, or linguistics.

A lightweight router, Gating Network a small neural network dynamically selects the most relevant experts for each input token. For example, the token "def" might activate coding-focused experts. Sparse Activation engages only top-k experts per token, drastically reducing computation compared to dense models. In the dense models all parameters are used per token.

## Architecture of DeepSeek-R1

```shell
Input Tokens → [Transformer Layers] → [MoE Layer] → [Transformer Layers] → Output  
                      │  
          (MoE Layer Breakdown)  
                      ├─── Expert 1 (FFN)  
                      ├─── Expert 2 (FFN)  
                      ├─── ...  
                      └─── Gating Network (Router)
```

The key efficiency of this architecture is activating less parameters per token. While the total model size is 671B parameters, only some small percentage of them are active per token, mimicking a smaller, faster model during inference.

## Innovations behing DeepSeek-R1

1. Auxiliary losses

These are supplementary loss terms added to the primary training objective to enforce balanced participation of experts. During training, the model is penalized if certain experts are consistently underutilized or overused. For example, an importance loss might compute the variance in expert selection frequency across a batch of data and penalize skew (e.g. if Expert 5 is chosen for 80% of tokens while others are neglected). Another variant, load loss, ensures each expert receives a roughly equal number of tokens over time by modeling expert selection as a “soft” distribution and minimizing deviations from uniformity. These losses prevent token collapse, a failure mode where the gating network overly relies on a small subset of experts, effectively wasting model capacity and degrading generalization. By incentivizing diversity in expert usage, auxiliary losses ensure specialized knowledge is distributed across the model.

2. Load balancing

This refers to algorithmic strategies that enforce equitable token-to-expert assignments during both training and inference. For instance, during training, the gating network’s logits (raw output scores before softmax) are adjusted to discourage repetitive expert selection. Techniques like expert buffering temporarily block overused experts from being selected until others catch up. Advanced implementations, such as router z-loss penalizing overly confident routing scores, smooth the gating distribution to avoid winner-takes-all scenarios. Load balancing also addresses hardware constraints: uneven expert utilization can cause GPU/TPU memory bottlenecks, as inactive experts still occupy VRAM. In frameworks like Google’s GShard or Meta’s Switch Transformer, load balancing is often achieved through top-k masking with noise—adding stochasticity to the gating process to ensure exploration of underused experts. The result is a self-correcting system where experts specialize in distinct patterns without monopolizing the workload, analogous to a well-managed team where tasks are dynamically delegated based on expertise and availability.

## Training Pipeline

1. Pre-training Phase

R1 trained on a diverse corpus large-scale dataset including web pages, academic papers, code repositories and multilingual content. Uses autoregressive language modeling predicting the next token in a sequence and/or masked language modeling predicting masked tokens in a sentence.

2. Fine-tuning Phase

The model is refined on structured tasks to enhance versatility using Multi-Task Learning.

- Reasoning: Benchmarks like GSM8K (grade-school math problems) and Big-Bench Hard (complex reasoning tasks).
- Coding: Datasets like HumanEval (Python code generation) and MBPP (short programming problems).
- General Knowledge: Tests like MMLU (Massive Multitask Language Understanding, covering 57 subjects).
- Curriculum Learning: Tasks are ordered from simple to complex (e.g. arithmetic → calculus → theorem proving), mirroring human educational progression.

3. Optimization Techniques

Gradient Checkpointing is used to reduce memory usage by recomputing intermediate activations during training instead of storing them.

## Performance and efficiency

Benchmark results are taken from their Github repo. It can process ~1,000 tokens/second on 8xA100 GPUs, enabled by MoE’s sparse activation.

![benchmark](/images/post_pics/deepseek-r1-model/benchmark.jpg)

### Trade-offs

1. MoE models require more VRAM to manage expert parameters, even when inactive.
2. Occasional inconsistencies between experts can cause erratic outputs, mitigated by balancing losses during training.

It's pretty exciting times on AI, let's see what future brings us :)
