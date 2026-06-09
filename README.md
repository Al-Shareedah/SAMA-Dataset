# SAMA: A Visual Question Answering Benchmark for Stylized Wayfinding Maps

[![Dataset](https://img.shields.io/badge/Dataset-Release-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#) 
<!-- [TODO: Update the badge links above with your actual URLs] -->

**SAMA** is the first large-scale, manually revised Visual Question Answering (VQA) dataset specifically designed to evaluate Vision-Language Model (VLM) spatial reasoning over stylized, non-standard indoor and outdoor wayfinding maps. 

Unlike standard cartographic maps, the attraction maps in SAMA (such as theme parks, zoos, and resorts) are not drawn to scale and lack standard geographic coordinates, presenting a unique and challenging benchmark for multimodal spatial intelligence.

## 📊 Dataset Overview

*   **Total Maps:** 49 real-world maps
*   **Total QA Pairs:** 4,296 ground-truth questions and answers
*   **Map Categories:** Theme Parks, Resorts, Zoos, Malls, Museums, and Trails
*   **Question Types:** Amenity search, symbol/legend interpretation, landmark identification, relative positioning, orientation, and localized navigation.
*   **Generation:** Hybrid LLM-assisted generation (Gemini 3 Pro / Gemma 3) with 100% human manual validation and revision.
