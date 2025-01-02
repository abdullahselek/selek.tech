---
title: "What Is Federated Learning?"
date: 2025-01-02T14:29:25+03:00
tags:
  - federated learning
  - horizontal federated learning
  - vertical federated learning
  - federated transfer learning
image: /images/post_pics/what-is-federated-learning/federated-learning.jpg
comments: true
---

Federated Learning (FL) is an advanced distributed machine learning paradigm that enables multiple clients such as devices, organizations, or data nodes—to collaboratively train a shared model while maintaining data privacy by keeping the data decentralized. In this approach, the raw data remains localized on each client's device or system, and only intermediate updates, such as gradients or model parameters, are communicated with a central server or aggregator. These updates are securely aggregated to refine and optimize a global model, thereby leveraging the collective intelligence of distributed datasets without violating data ownership or confidentiality. This innovative framework is particularly crucial in scenarios where data sensitivity, security regulations, or logistical constraints prevent the centralization of data. By bridging the gap between privacy-preserving technologies and collaborative intelligence, federated learning has emerged as a cornerstone in the development of ethical and scalable artificial intelligence solutions.

## Key Characteristics of Federated Learning

1. **Privacy-Preserving** Sensitive data remains local, reducing privacy risks.
2. **Decentralization** Data is distributed across multiple devices or organizations.
3. **Collaborative** Participants contribute to training a shared model without sharing their data.
4. **Communication-Efficient** Optimizations minimize the communication cost between clients and the central server.

## How does Federated Learning work?

FL operates through a decentralized training process that involves multiple clients and a central aggregator or server. Initially, a global model is shared with participating clients, such as mobile devices or organizational servers. Each client uses its local data to train the model independently, performing computations such as gradient updates or parameter adjustments. Instead of sharing raw data, the clients securely transmit their locally computed model updates back to the central server. The server then aggregates these updates, often using techniques like weighted averaging, to refine the global model. This updated global model is redistributed to clients, initiating a new training round. The iterative process continues until the model converges to a satisfactory performance level. Throughout this workflow, FL ensures that sensitive data remains local, addressing privacy concerns, reducing communication costs, and enabling collaborative learning across distributed and heterogeneous datasets.

## Types of Federated Learning

FL can be categorized into three main types based on the distribution of data and the nature of collaboration between clients. These types are designed to address different real-world data distribution scenarios and privacy concerns.

1. Horizontal Federated Learning (HFL)

Horizontal FL is applied when the datasets of different participants (clients) share the same features but differ in the samples (i.e., data points). As an example, two hospitals in different regions have patient records with similar attributes (e.g., age, diagnosis, treatment) but for different patients. Clients have datasets with the same feature space but different sample IDs. Ideal for scenarios where participants operate in similar domains.

Use Cases of HFL, Federated training of ML models across multiple edge devices (e.g., smartphones) and collaborative learning among organizations within the same industry.

2. Vertical Federated Learning (VFL)

Vertical FL is used when datasets from different clients have overlapping sample IDs but differ in feature sets. Clients have datasets with overlapping sample IDs but distinct feature spaces. Secure alignment is necessary to identify shared samples between participants.

Cross-industry collaborations, such as combining healthcare and insurance data and enhancing models using complementary features from different organizations are some of the use cases.

3. Federated Transfer Learning (FTL)

Federated Transfer Learning is used when the datasets of participants differ in both sample IDs and feature sets. It leverages transfer learning techniques to enable collaboration. Data is disjoint in both feature and sample spaces. Transfer learning techniques adapt the pre-trained model to a new domain.

Collaboration between organizations in completely different domains and building models for applications with sparse or limited data.

### Comparison of Federated Learning Types

| **Type**              | **Feature Space** | **Sample Space** | **Use Case**                              |
|------------------------|-------------------|------------------|-------------------------------------------|
| **Horizontal FL (HFL)**    | Same              | Different        | Similar organizations, distributed devices|
| **Vertical FL (VFL)**      | Different         | Same             | Cross-industry collaborations             |
| **Federated Transfer Learning (FTL)** | Different         | Different        | Sparse or non-overlapping data scenarios |

### Hybrid Federated Learning

In real-world applications, scenarios often do not strictly fall into one type. Hybrid FL combines elements of HFL, VFL, and FTL to address complex data distribution and collaboration needs.

## Importance of Federated Learning in Machine Learning

1. **Enhanced Privacy and Security** FL minimizes privacy concerns as raw data is not transferred, making it suitable for applications involving sensitive data like healthcare or financial records.
2. **Regulatory Compliance** FL aligns with regulations like GDPR and HIPAA by ensuring data remains on-premises or within regions.
3. **Scalable Learning** By utilizing distributed client resources, FL can leverage vast amounts of data without centralized storage.
4. **Edge Computing Integration** FL is integral to edge computing, enabling real-time AI applications like autonomous vehicles, IoT, and personalized mobile services.
5. **Better Model Generalization** Training on diverse datasets from multiple clients can result in more robust models that generalize well to different environments.

## Why Use Federated Learning?

1. **Data Privacy and Security** Addresses privacy concerns by keeping data on local devices.
2. **Access to Diverse Data Sources** Enables collaboration across organizations or devices with data silos.
3. **Cost Efficiency** Reduces the need for central data storage and transfer costs.
4. **Compliance with Laws and Ethics** Helps organizations adhere to data protection laws while leveraging ML.
5. **Real-Time AI Applications** Supports scenarios where data is generated and processed on edge devices.

## Top 3 Federated Learning Frameworks

1. [Flower](https://flower.ai/)

Flower is framework-agnostic and easy to integrate with PyTorch, TensorFlow, etc. Also it is flexible for research and production and also supports cross-silo and cross-device FL.

2. [PySyft](https://github.com/OpenMined/PySyft)

PySyft is focused on privacy-preserving machine learning and also supports FL with secure multi-party computation (SMPC) and differential privacy. Good for research and prototyping.

3. [FATE](https://github.com/FederatedAI/FATE)

Designed for enterprise applications and supports secure computation techniques like homomorphic encryption. It is also robust for industrial FL use cases.

## Conclusion

Federated Learning represents a paradigm shift in machine learning, enabling collaborative intelligence while addressing critical concerns of data privacy, security, and compliance. By decentralizing the training process, FL ensures that sensitive data remains local, empowering industries and individuals to harness the power of AI without compromising confidentiality.
