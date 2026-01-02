# NLP Question Answering Retrieval System

This project explores different information retrieval techniques for Question Answering tasks using classical, neural, and hybrid models.

## Overview
- Implements **BM25**, **Dense Passage Retrieval (DPR)**, and a **Hybrid (BM25 + DPR)** model
- Uses **FAISS** for fast vector similarity search
- Evaluates models on datasets like **COVID-QA** and **Space/Wikipedia**
- Compares models based on **accuracy** and **latency**
- Includes a simple frontend/demo for testing queries

## Technologies Used
- Python
- Hugging Face Transformers
- FAISS
- PyTorch
- Rank-BM25
- Datasets (Hugging Face)

## How It Works
1. Load and preprocess datasets
2. Index documents using BM25 and DPR encoders
3. Retrieve top documents for a given question
4. Evaluate performance and compare results

## Results
The hybrid model generally balances accuracy and speed better than using BM25 or DPR alone.

## Setup
```bash
pip install faiss-cpu rank_bm25 datasets transformers torch
