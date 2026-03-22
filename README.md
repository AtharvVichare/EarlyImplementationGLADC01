# RnD Olympics 2020: Graph Preprocessing Pipeline

[cite_start]This repository contains the preprocessing workflow for the **RnD Olympics Dataset 2020**[cite: 1]. [cite_start]The pipeline transforms raw, tabular particle physics data into graph-structured objects suitable for Geometric Deep Learning (GDL)[cite: 2, 46].

---

## 1. Dataset Overview
[cite_start]The dataset consists of simulated high-energy physics events used to train models for anomaly detection and event classification[cite: 1, 30].

* **Total Events**: 1.1 million total events.
* [cite_start]**Standard Model (SM)**: 1,000,000 events used for baseline training[cite: 10, 11, 30].
* [cite_start]**Beyond Standard Model (BSM)**: 100,000 events[cite: 10, 12].

---

## 2. Data Structure
[cite_start]The raw dataset is stored in a structured tabular format where each row represents an entire event[cite: 14, 23].

* [cite_start]**Event Capacity**: Each row (event) contains data for up to **700 particles**[cite: 14, 16].
* [cite_start]**Particle Features**: Each particle is defined by three kinematic variables: transverse momentum ($p_T$), pseudorapidity ($\eta$), and azimuthal angle ($\phi$)[cite: 14, 45].
* [cite_start]**Padding**: Events with fewer than 700 particles are filled with **zero padding** to maintain consistent row length[cite: 28, 29].

---

## 3. Preprocessing Workflow
[cite_start]The transformation from raw rows to graphs follows these sequential steps[cite: 27]:

### Step 1: Extraction & Filtering
* [cite_start]**Extraction**: Particles are extracted from the raw dataset on an event-by-event basis[cite: 33, 34].
* [cite_start]**Zero-Padding Removal**: All $(0, 0, 0)$ entries are stripped during extraction to isolate actual particle hits[cite: 28, 29].

### Step 2: Jet Clustering
* [cite_start]**Algorithm**: The **anti-$k_T$** clustering algorithm is used to group particles into jets[cite: 35, 36].
* **Parameters**: Radius parameter $R = 1$[cite: 37].
* [cite_start]**Tool**: Implementation via the `pyjet` library[cite: 37].

### Step 3: Graph Construction
* [cite_start]**Algorithm**: A graph is built for each jet using the **k-Nearest Neighbors (k-NN)** algorithm[cite: 39, 40].
* **Connectivity**: Each node is connected to its $k=8$ nearest neighbors[cite: 42].

---

## 4. Output Representation
The final processed data is wrapped as a **PyTorch Geometric (PyG)** `Data` object[cite: 46]:

* [cite_start]**Node Features (`x`)**: A matrix containing $[p_T, \eta, \phi]$ for each particle[cite: 45].
* **Edge Index (`edge_index`)**: A graph connectivity matrix in COO format[cite: 46].
* **Labels (`y`)**: Truth labels for event classification[cite: 45, 46].

---

> **Note**: This pipeline is optimized for training on Standard Model (SM) backgrounds while remaining compatible with BSM signal testing[cite: 7, 30].
