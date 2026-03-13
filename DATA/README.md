# Data

The Sentinel-1 GRD imagery used in this thesis was accessed via the 
Google Earth Engine platform. No data files are included in this 
repository as the `.tif` files are too large to host on GitHub.

## How to reproduce the dataset

1. Open the Google Earth Engine code editor at https://code.earthengine.google.com
2. Copy and run the script in `GEE/export_sentinel1.js`
3. The script exports a multi-band GeoTIFF to your Google Drive, 
   with one band per acquisition per polarisation, named VV_YYYYMMDD / VH_YYYYMMDD
4. Download the exported file from Google Drive and update the 
   `TIF_PATH` variable in `REAL_DATA/flood_pipeline.py` to point to it

## Study region
- Region of interest: (-0.7819, 38.0828) to (-0.7359, 38.1288) 
  (longitude, latitude, decimal degrees)
- Monitoring window: 22 September 2018 to 11 September 2020
- Acquisition constraints: IW swath mode, dual polarisation (VV and VH), 
  descending passes, relative orbit 110
