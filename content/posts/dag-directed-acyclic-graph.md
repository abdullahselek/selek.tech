---
title: "What is DAG - Directed Acyclic Graph?"
date: 2025-01-01T15:41:43+03:00
tags: ["dag", "directed acyclic graph", "ml pipelines", "airflow", "workflow orchestration"]
image: /images/post_pics/dag-directed-acyclic-graph/direct-acyclic-graph.jpg
comments: false
---

Happy New Year! 2024 was a busy year for me and I could finally find some time to write. In this post I'll try to share some details about DAGs and their usage in ML pipelines. Let's move into the subject.

DAGs are a foundational tool in ML and data workflows, helping researchers and engineers organize and optimize complex processes effectively. But what is DAG? A DAG (Directed Acyclic Graph) in Machine Learning (ML) is a conceptual framework used to represent workflows, data pipelines, or computational processes. It consists of nodes (tasks, operations, or computations) and directed edges (dependencies between these tasks).

**Nodes** signify distinct tasks, computations, or data operations, such as loading datasets, preprocessing features, training models, or evaluating outputs. On the other hand **Edges** represent dependencies between these tasks, indicating that one operation must be completed before another begins. DAG doesn't have any loops there’s no way to start from a node and return to it by following the directed edges. This ensures a clear, sequential progression of tasks.

## Key Characteristics of a DAG

A DAG is both **Directed** and **Acyclic** meaning each edge has a direction, representing the flow of data or computation and the graph has no cycles, meaning you cannot return to a node by following the directed edges.

![transitive-closure](/images/post_pics/dag-directed-acyclic-graph/dag-transitive-closure.jpg)

## How a DAG Operates?

A DAG operates by organizing tasks or operations as nodes connected by directed edges that define dependencies, ensuring a clear execution order. Tasks without dependencies are executed first, followed by dependent tasks as their prerequisites are completed. This structure allows for topological sorting, which determines the proper sequence of execution while preventing cycles that could cause infinite loops. Independent tasks can run in parallel, enhancing efficiency, while edges facilitate the flow of data or results between tasks. In case of failure, only the affected task and its dependents are paused or retried, ensuring targeted error handling without disrupting unrelated parts of the workflow. This modular and dependency driven approach makes DAGs ideal for managing complex processes like machine learning pipelines or data workflows.

![topological-sort-order](/images/post_pics/dag-directed-acyclic-graph/dag-topological-sort-order.jpg)

## Applications of DAGs in Machine Learning

1. Data Processing Pipelines

DAGs are commonly used to structure the flow of data from initial ingestion through processing, model training, and deployment stages. For example, Apache Airflow uses DAGs to manage ETL (Extract, Transform, Load) workflows.

2. Computational Graphs in Deep Learning

In frameworks like PyTorch and TensorFlow, DAGs represent models as a series of computations. Nodes correspond to operations (e.g., matrix multiplications, activation functions), and edges define how data flows between them. This structure enables efficient computation, particularly during forward and backward propagation.

3. Causal Inference

DAGs are used to model cause and effect relationships within a system. Nodes represent variables, while edges signify causal dependencies, helping researchers analyze interventions and their outcomes.

4. Workflow Automation

In ML pipelines, DAGs manage task dependencies, automating processes like data cleaning, feature engineering, model training, and evaluation. This ensures tasks are executed in the correct sequence.

5. Federated Learning

In distributed systems like federated learning, DAGs can model sequential tasks, such as aggregating updates from client devices and refining the global model.

6. Bayesian Networks

DAGs are at the heart of probabilistic models, particularly Bayesian networks, where they define random variables (nodes) and their conditional dependencies (edges), offering a framework for reasoning under uncertainty.

## Why DAGs Matter?

- DAGs enable dependency management, they ensure that tasks are executed in the correct order, respecting all dependencies.
- They support parallelism so that independent tasks can run simultaneously, improving efficiency.
- DAGs give clarity. Visually and conceptually simplify complex workflows, making them easier to understand and debug.
- Enable reusability, their modular design allows components to be reused across different projects or experiments.

## Workflow Orchestration Tools

1. [Apache Airflow](https://airflow.apache.org/)

A platform for programmatically authoring, scheduling, and monitoring workflows as DAGs. Widely used in data engineering and ML pipelines for task automation.

2. [Luigi](https://luigi.readthedocs.io/en/stable/index.html)

A Python based workflow management tool that uses DAGs to manage task dependencies and build complex pipelines.

3. [Prefect](https://www.prefect.io/)

A workflow orchestration tool designed for data workflows and ML pipelines, offering a user friendly API for defining DAGs and monitoring executions.

4. [Dagster](https://dagster.io)

A modern orchestration platform for building and managing data pipelines, with native support for defining and executing DAGs.

5. [Nextflow](https://www.nextflow.io)

A bioinformatics oriented workflow management system that uses DAGs to process and analyze data, particularly in genomics.

## Summary

DAGs are foundational structures in computational and data workflows, providing a clear and efficient way to model dependencies and manage complex processes. By organizing tasks or operations into nodes connected by directed edges, DAGs ensure tasks are executed in the correct sequence, avoid cyclic dependencies, and enable parallelism where possible. This makes them invaluable in fields like machine learning, workflow orchestration, data engineering, and causal inference. Tools and frameworks that leverage DAGs streamline the development, execution, and optimization of workflows, enhancing transparency, scalability, and error handling. The importance of DAGs lies in their ability to bring order to complexity, making them indispensable for modern computational systems and scalable data driven applications.
