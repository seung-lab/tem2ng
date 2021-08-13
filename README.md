# tem2ng
Convert raw TEM microscope images into a neuroglancer volume.

```bash
# approximate usage 
tem2ng info matrix://BUCKET/LAYER --dataset-size 10000,10000,1 --resolution 4,4,40 --chunk-size 1000,1000,1
tem2ng upload subtiles matrix://BUCKET/LAYER
```