# RnD Olympics 2020: Graph Preprocessing Pipeline

This repository contains the preprocessing workflow for the **RnD Olympics Dataset 2020**. The pipeline transforms raw, tabular particle physics data into graph-structured objects suitable for Geometric Deep Learning (GDL).

---

## 1. Dataset Overview
The dataset consists of simulated high-energy physics events used to train models for anomaly detection and event classification.

* **Total Events**: 1.1 million total events.
* **Standard Model (SM)**: 1,000,000 events used for baseline training.
* **Beyond Standard Model (BSM)**: 100,000 events.

---

## 2. Data Structure
The raw dataset is stored in a structured tabular format where each row represents an entire event.

* **Event Capacity**: Each row (event) contains data for up to **700 particles**.
* **Particle Features**: Each particle is defined by three kinematic variables: transverse momentum ($p_T$), pseudorapidity ($\eta$), and azimuthal angle ($\phi$).
* **Padding**: Events with fewer than 700 particles are filled with **zero padding** to maintain consistent row length.

---

## 3. Preprocessing Workflow
The transformation from raw rows to graphs follows these sequential steps:

### Step 1: Extraction & Filtering
* **Extraction**: Particles are extracted from the raw dataset on an event-by-event basis.
* **Zero-Padding Removal**: All $(0, 0, 0)$ entries are stripped during extraction to isolate actual particle hits.

### Step 2: Jet Clustering
* **Algorithm**: The **anti-$k_T$** clustering algorithm is used to group particles into jets.
* **Parameters**: Radius parameter $R = 1$.
* **Tool**: Implementation via the `pyjet` library.

### Step 3: Graph Construction
* **Algorithm**: A graph is built for each jet using the **k-Nearest Neighbors (k-NN)** algorithm.
* **Connectivity**: Each node is connected to its $k=8$ nearest neighbors.

---

## 4. Output Representation
The final processed data is wrapped as a **PyTorch Geometric (PyG)** `Data` object:

* **Node Features (`x`)**: A matrix containing $[p_T, \eta, \phi]$ for each particle.
* **Edge Index (`edge_index`)**: A graph connectivity matrix in COO format.
* **Labels (`y`)**: Truth labels for event classification.

---

> **Note**: This pipeline is optimized for training on Standard Model (SM) backgrounds while remaining compatible with BSM signal testing.
