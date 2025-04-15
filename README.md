# RAG-for-RPMS

## Retrieval-Augmented Question Answering (RAG) System

A simple but effective Retrieval-Augmented Generation (RAG) system using TF-IDF for document retrieval and the T5 transformer model for answer generation.

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that:

    Retrieves the most relevant document from a corpus using TF-IDF and cosine similarity.

    Uses a pre-trained T5 model to generate an answer based on the retrieved document.

The system is capable of answering natural language questions with context-aware responses.


## How It Works

    Preprocessing: Text documents are cleaned, lowercased, and tokenized.

    Vectorization: TF-IDF vectorizer is used to convert documents into numerical vectors.

    Retrieval: Cosine similarity is used to retrieve the most relevant document for a given question.

    Answer Generation: A T5 model generates an answer using the format:


 ## Evaluation

Basic evaluation is performed using F1-score between generated answers and reference answers. Current model performance:

    F1 Score: 0.47
