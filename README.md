# SAMA: A Visual Question Answering Benchmark for Stylized Wayfinding Maps

[![Dataset](https://img.shields.io/badge/Dataset-Release-blue.svg)](https://github.com/Al-Shareedah/SAMA-Dataset/tree/main/data)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Al-Shareedah/SAMA-Dataset/blob/main/LICENSE)


**SAMA** is the first large-scale, manually revised Visual Question Answering (VQA) dataset specifically designed to evaluate Vision-Language Model (VLM) spatial reasoning over stylized, non-standard indoor and outdoor wayfinding maps. Developed at the University of California, Riverside, this dataset pushes the boundaries of spatial comprehension in AI.

Unlike standard cartographic maps, the attraction maps in SAMA (such as theme parks, zoos, and resorts) are not drawn to scale and lack standard geographic coordinates, presenting a unique and challenging benchmark for multimodal spatial intelligence.

This repository contains the dataset resources corresponding to our short paper:
📄 **[SAMA: Spatial Question Answering over Maps of Attractions [Experiments]]**

## 📊 Dataset Overview

*   **Total Maps:** 49 real-world maps
*   **Total QA Pairs:** 4,296 ground-truth questions and answers
*   **Map Categories:** Theme Parks, Resorts, Zoos, Malls, Museums, and Trails
*   **Question Types:** Amenity search, symbol/legend interpretation, landmark identification, relative positioning, orientation, and localized navigation.
*   **Generation:** Hybrid LLM-assisted generation (Gemini 3 Pro / Gemma 3) with 100% human manual validation and revision.

## 🗂️ Data Format

The question-answer pairs are organized by map category in JSON format. Each entry contains the necessary metadata to map the question to its corresponding image. 

Here is an example of the data structure from `mall_questions.json`:

```json
{
  "dataset": "SAMA",
  "category": "mall",
  "question_count": 100,
  "questions": [
    {
      "question_id": "SAMA_MALL_COASTAL_GALLERIA_MALL_Q0001",
      "image_id": "SAMA_MALL_COASTAL_GALLERIA_MALL",
      "image_filename": "coastal_galleria.png",
      "question": "How many Clothing stores are there in the mall?",
      "reference_answers": [
        "10.0"
      ]
    },
    {
      "question_id": "SAMA_MALL_COASTAL_GALLERIA_MALL_Q0002",
      "image_id": "SAMA_MALL_COASTAL_GALLERIA_MALL",
      "image_filename": "coastal_galleria.png",
      "question": "In which map direction is Swarovski located relative to Sushi Siam?",
      "reference_answers": [
        "Southwest"
      ]
    }
  ]
}
