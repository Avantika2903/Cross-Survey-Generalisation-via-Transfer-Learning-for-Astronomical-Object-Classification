# Cross-Survey-Generalisation-via-Transfer-Learning-for-Astronomical-Object-Classification

Overview

This project investigates the effectiveness of cross-survey transfer learning for astronomical object classification using deep learning.

Modern astronomical sky surveys, such as the Sloan Digital Sky Survey (SDSS) and the Dark Energy Spectroscopic Instrument (DESI), observe the same celestial objects but differ in image quality, instrumentation, observational conditions, and data distributions. These differences introduce a domain shift, making it difficult for models trained on one survey to perform well on another.

The objective of this project is to study how knowledge learned from the SDSS dataset can be transferred to the DESI dataset using different transfer learning strategies.

The classification task focuses on three astronomical object classes:

 Galaxy
 
 Star
 
 Quasar (QSO)

 Datasets
 
SDSS (Sloan Digital Sky Survey)

Source of pretrained knowledge

Images retrieved using SQL queries through SDSS SkyServer

Classes:

Galaxy

Star

Quasar (QSO)

DESI (Dark Energy Spectroscopic Instrument)

Target dataset for transfer learning

Used for fine-tuning and evaluation

Methodology

The proposed workflow consists of the following stages:

Data collection from SDSS and DESI

Image preprocessing

Baseline model training

Cross-survey evaluation

Transfer learning strategies  using EfficientNet-B0

Performance evaluation
Target dataset for transfer learning
Used for fine-tuning and evaluation
