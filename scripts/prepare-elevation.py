"""Prepare NOAA ETOPO elevation (meters) for rendering; requires scipy once.

Download:
https://oceanwatch.pifsc.noaa.gov/erddap/griddap/ETOPO_2022_v1_60s.nc?z[0:4:10799][0:4:21599]
No scipy dependency is needed by the production renderer.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.io import netcdf_file

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('source',type=Path)
p.add_argument('destination',type=Path)
a=p.parse_args()
with netcdf_file(a.source,mmap=False) as ds:
    heights=ds.variables['z'].data.astype(np.float32)
    lat=ds.variables['latitude'].data.copy()
    lon=ds.variables['longitude'].data.copy()
assert heights.shape==(2700,5400)
assert np.isfinite(heights).all() and heights.min()>-15000 and heights.max()<10000
if lat[0]<lat[-1]: heights=heights[::-1]
start=int(np.searchsorted(lon,180.0))
heights=np.roll(heights,-start,axis=1)
heights=np.maximum(heights,0).round().astype(np.uint16)
checks={}
for name,la,lo,minimum,maximum in [('Himalaya',28,86.9,3000,9000),('Tibetan Plateau',31,88,3500,8000),('Pacific',0,-140,0,0)]:
    row=round((90-la)/180*(heights.shape[0]-1)); col=int((lo+180)/360*heights.shape[1])
    value=int(heights[row,col]); assert minimum<=value<=maximum,(name,value)
    checks[name]=value
Image.fromarray(heights).save(a.destination,optimize=True)
print(json.dumps({'size':list(heights.shape[::-1]),'units':'meters','checks':checks,'maximum':int(heights.max()),'file':str(a.destination)}))
