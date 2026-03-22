# Data-Preprocessing-for-Rnd-LHC-Olympics-Dataset
This project provides a specialized preprocessing pipeline for the RnD Olympics Dataset 2020, designed to transform raw particle physics event data into graph-structured formats suitable for GNN. By leveraging jet clustering and k-Nearest Neighbors (k-NN) graph construction.
Project Overview: RnD Olympics 2020 Graph Preprocessing
This repository contains a pipeline to preprocess the RnD Olympics 2020 dataset, specifically focusing on converting event-level particle data into PyTorch Geometric (PyG) data objects.
Dataset Specifications
Total Events: 1.1 million events (1,000,000 SM and 100,000 BSM).
Data Models: Includes both Standard Model (SM) and Beyond Standard Model (BSM) physics.
Input Structure: Each row in the raw dataset represents a single whole event.
Particle Capacity: Each event row contains data for up to 700 particles, including transverse momentum ($p_T$), pseudorapidity ($\eta$), azimuthal angle ($\phi$), and mass ($m$).
Preprocessing Pipeline
The pipeline follows a structured four-step process to prepare data for model training:
1.Event-Wise Extraction: Individual particles are extracted from the raw 1D row format.
2.Padding Removal: All zero-padded entries used for row-length consistency are filtered out.
3.Jet Clustering: Particles are clustered into jets using the anti-$k_T$ algorithm with a radius of $R=1$ via the pyjet library.
4.Graph Construction: For each clustered jet, a graph is generated using the k-NN algorithm with a connectivity of $k=8$.
Output Format
The final processed output is wrapped as a PyTorch Geometric Data object with the following attributes:
x: Node feature matrix containing particle kinematics ($p_T, \eta, \phi$).
edge_index: Graph connectivity in COO format.
y: Event labels.
