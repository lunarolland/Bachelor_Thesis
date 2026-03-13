# Online Changepoint Detection for Real-Time Flood Event Identification

Bachelor Thesis — Luna Rolland, École Polytechnique  
Supervisor: Dr Dean Bodenham, Imperial College London  
January 2026 – March 2026

## Overview

This repository contains the code used in my bachelor thesis, which investigates 
the use of online changepoint detection methods for identifying flood events from 
Sentinel-1 SAR satellite imagery. A CUSUM-based monitoring pipeline is developed 
and applied to a flood event that occurred in southeastern Spain in September 2019.

## Repository Structure

- `CUSUM/` — Core CUSUM implementation and burn-in variant
- `SIMULATION/` — Synthetic flood simulations and parameter sweep
- `REAL_DATA/` — Main monitoring pipeline and BCP benchmark
- `GEE/` — Google Earth Engine script to export the Sentinel-1 data
- `DATA/` — Instructions for reproducing the dataset

## How to Reproduce the Results

1. Follow the instructions in `DATA/README.md` to obtain the Sentinel-1 GeoTIFF
2. Install dependencies: `pip install -r requirements.txt`
3. Run the parameter sweep: `python SIMULATION/test_CUSUM.py`
4. Run the main pipeline: `python REAL_DATA/flood_pipeline.py`
5. Run the BCP benchmark: `python REAL_DATA/flood_paper.py`

## Dependencies

See `requirements.txt`. Main dependencies are `numpy`, `pandas`, 
`matplotlib`, `rasterio`, and `rpy2` (for the BCP benchmark, 
which also requires the R package `bcp`).
