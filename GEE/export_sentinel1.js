// ---- 0) CHIP GEOMETRY 
var chip = ee.Geometry.Rectangle([
  -0.7818936232976315, 38.08281882884935,
  -0.7358998807507120, 38.12881257139627
]);

Map.centerObject(chip, 13);
Map.addLayer(chip, {color: 'red'}, 'Sen1Floods11 chip');

// ---- 1) FLOOD DATE + ±1 YEAR WINDOW ----
var floodDate = ee.Date('2019-09-17');
var start = floodDate.advance(-1, 'year');
var end   = floodDate.advance( 1, 'year');

print('Flood date:', floodDate);
print('Window start:', start);
print('Window end:', end);

// ---- 2) SENTINEL-1 TIME SERIES (VV/VH) ----
// Match event metadata: DESCENDING, relative orbit 110
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(chip)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .filter(ee.Filter.eq('relativeOrbitNumber_start', 110))
  .select(['VV', 'VH']);

var s1_ts = s1.filterDate(start, end);

print('S1 images (DESC + relOrbit 110) in ±1y window:', s1_ts.size());
print('First image date:', ee.Date(s1_ts.sort('system:time_start').first().get('system:time_start')));
print('Last image date:', ee.Date(s1_ts.sort('system:time_start', false).first().get('system:time_start')));

// ---- 3) VISUALIZATION  ----
var vvVis = {min: -25, max: 0, palette: ['000000', 'ffffff']};
var vhVis = {min: -35, max: -5, palette: ['000000', 'ffffff']};

// ---- 4) STACK TIME SERIES INTO MULTIBAND IMAGE + EXPORT ----
var s1_stack = s1_ts.toBands().clip(chip);

Export.image.toDrive({
  image: s1_stack,
  description: 'Spain_7370579_S1_VV_VH_relOrbit110_DESC_pm1y',
  folder: 'gee_exports',
  fileNamePrefix: 'Spain_7370579_S1_VV_VH_relOrbit110_DESC_pm1y',
  region: chip,
  scale: 10,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});
