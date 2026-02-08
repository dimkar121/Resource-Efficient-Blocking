# Resource-Efficient Blocking: Optimizing the Trade-off between Effectiveness and Scalability in Entity Resolution

This repository contains the source code and datasets for the paper: "Resource-Efficient Blocking: Optimizing the Trade-off between Effectiveness and Scalability in Entity Resolution" by D. Karapiperis (International Hellenic University), C. Tjortjis (International Hellenic University), and V.S. Verykios (Hellenic Open University).

## 📖 Abstract

Deep Learning has revolutionized Entity Resolution (ER) by enabling high-accuracy matching through dense vector embeddings. However, this paradigm shift transforms the traditional blocking step into a high-dimensional similarity search problem, introducing a massive computational bottleneck that threatens scalability. While modern ER frameworks increasingly rely on blocking, the so-called indexing, to generate candidate pairs, the architectural design of this retrieval stage is often treated as a black box, with little attention paid to the resource implications of high-dimensional models. This paper conducts a rigorous design space exploration of blocking architectures, evaluating nine candidate generation strategies across graph-based, partition-based, and hash-based families. A primary contribution is a systematic analysis of Vector Quantization—specifically Product Quantization and Scalar Quantization—demonstrating that quantization is not merely a memory optimization, but a critical requirement for deploying state-of-the-art embeddings at scale. We evaluate these architectures on seven real-world datasets, measuring the interplay between embedding granularity, recall, and computational cost. Our results reveal that while graph-based methods  offer peak precision, quantized partitioning methods reduce memory footprints by up to 96\% with negligible loss in recall, effectively democratizing high-performance ER on commodity hardware. The study culminates in a decision framework for designing scalable, resource-aware blocking pipelines.

## 🚀 Key Contributions

* **Rigorous Benchmarking:** An extensive experimental comparison of nine state-of-the-art ANN algorithms across eight diverse datasets.
* **In-depth Quantization Analysis:** A detailed quantitative analysis of the impact of Product Quantization (PQ) and Scalar Quantization (SQ) on search accuracy, query latency, memory footprint, and index construction time.
* **Clear Performance Hierarchy:** Identification of the most effective and robust algorithms, highlighting the strengths of graph-based and IVF-based methods and the limitations of LSH.
* **Actionable Recommendations:** A practical decision framework to guide the selection of the most appropriate ANN indexing strategy based on specific application constraints.

## 🛠️ Methods Evaluated

The evaluation covers nine ANN algorithms from the most influential families:

1.  **HNSW (Hierarchical Navigable Small World)**
2.  **IVF (Inverted File)**
3.  **ANNOY (Approximate Nearest Neighbors Oh Yeah)**
4.  **SCANN (Scalable Nearest Neighbors)**
5.  **Cosine LSH (Locality-Sensitive Hashing)**
6.  **HNSWPQ** (HNSW with Product Quantization)
7.  **HNSWSQ** (HNSW with Scalar Quantization)
8.  **IVFPQ** (IVF with Product Quantization)
9.  **IVFSQ** (IVF with Scalar Quantization)

## 📊 Datasets

The experiments were conducted on a diverse suite of nine real-world and semi-synthetic datasets:

* **Product Matching:** ABT-BUY, AMAZON-WALMART, AMAZON-GOOGLE, WDC
* **Bibliographic Matching:** ACM-DBLP, SCHOLAR-DBLP
* **Movies matching:** IMDB-DBPEDIA
* **Large-Scale Synthetic:** DBLP, VOTERS

All experiments were run using two different sentence-transformer models for embedding generation: `MiniLM-L6-v2` (Mini) and `Microsoft E5-large-v2` (E5).

## ⚙️ Setup and Installation

The implementations rely on several key open-source libraries. You can install them using pip:
 ```bash
    pip install -r requirements.txt
 ```

You can install a CUDA-enabled library for FAISS, be aware though of a potential incompatibility between `faiss-gpu` and `scann` concerning their required **NumPy** versions.
If the library of FAISS that supports your CUDA requires an older version of NumPy (e.g., `<2.0`), then SCANN, which may require a newer version (e.g., `>=2.0`), should be installed and run in a separate virtual environment. This will prevent package version conflicts and ensure both libraries function correctly for the experiments.

   
## ▶️ Running the Experiments

The repository is structured to allow for easy replication of the results presented in the paper.

1.  **Embedding Generation:** Use the provided scripts `embed_<DATASET>.py --model Mini` and `embed_<DATASET>.py --model E5` to generate the Mini and E5 embeddings (in terms of .pqt files) for each dataset, e.g., `embed_SCHOLAR-DBLP.py --model Mini`. For ABT-BUY and ACM-DBLP, we provide the corresponding .pqt files with the Mini embeddings. The large scale datasets—WDC, DBLP, and VOTERS—can be found [here](https://drive.google.com/drive/folders/1IM9Ot8zpx11YcwXe_4ZTVeEx6wFaHiOo?usp=sharing).

3.  **Running a Single Experiment:** You can run the evaluation for a specific dataset using the main script. For example:
    ```bash
    python experiments.py --model Mini --dataset ABT-BUY 
    ```



