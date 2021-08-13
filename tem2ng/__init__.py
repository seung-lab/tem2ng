import re
import os

import click
from tqdm import tqdm

from cloudvolume import CloudVolume, Bbox
from cloudvolume.exceptions import InfoUnavailableError
import numpy as np
import tinybrain

import cv2

TILE_REGEXP = re.compile(r'tile_(\d+)_(\d+)\.bmp')

def get_ng(tilename, z=0):
	x_map = {4:0,5:1,6:2,3:0,0:1,7:2,2:0,1:1,8:2}
	get_x = lambda t1,t2: 6000 * ((t1%24)*3 + x_map[t2])
	y_map = {4:0,3:1,2:2,5:0,0:1,1:2,6:0,7:1,8:2}
	get_y = lambda t1,t2: 6000 * ((t1//24)*3 + y_map[t2])

	t1, t2 = [ int(_) for _ in re.match(TILE_REGEXP, tilename).groups() ]
	
	x0 = get_x(t1, t2)
	xf = x0 + 6000
	y0 = get_y(t1,t2)
	yf = y0 + 6000

	return f"{x0}-{xf}_{y0}-{yf}_{z}-{z+1}"

class Tuple3(click.ParamType):
  """A command line option type consisting of 3 comma-separated integers."""
  name = 'tuple3'
  def convert(self, value, param, ctx):
    if isinstance(value, str):
      try:
        value = tuple(map(int, value.split(',')))
      except ValueError:
        self.fail(f"'{value}' does not contain a comma delimited list of 3 integers.")
      if len(value) != 3:
        self.fail(f"'{value}' does not contain a comma delimited list of 3 integers.")
    return value
  

@click.group()
def main():
	"""
	Upload tiles from a specific setup at PNI to
	cloud storage.
	"""
	pass

@main.command()
@click.option('--dataset-size', type=Tuple3(), default=None, required=True, help="Dimensions of the dataset in voxels.")
@click.option('--voxel-offset', type=Tuple3(), default="0,0,0", help="Dimensions of the dataset in voxels.")
@click.option('--chunk-size', type=Tuple3(), default="1024,1024,1", help="Chunk size of new layers.", show_default=True)
@click.option('--resolution', type=Tuple3(), default="1,1,1", help="Resolution of a layer in nanometers.", show_default=True)
@click.option('--bit-depth', type=int, default=8, help="Resolution of a layer in nanometers.", show_default=True)
@click.option('--num-mips', type=int, default=1, help="Number of mip levels to generate at once.", show_default=True)
@click.argument("cloudpath")
def info(
	cloudpath, 
	dataset_size, voxel_offset, chunk_size, 
	resolution, bit_depth, num_mips
):
	"""
	Creates and uploads the neuroglancer info file.
	This defines the size and properties of the image.
	We assume a grayscale image for microscopes.
	"""
	if bit_depth not in (8,16,32,64):
		print("tem2ng: bit depth must be 8, 16, 32, or 64.")

	info = CloudVolume.create_new_info(
	    num_channels    = 1,
	    layer_type      = 'image',
	    data_type       = f'uint{bit_depth}',
	    encoding        = 'raw', 
	    resolution      = resolution, # Voxel scaling, units are in nanometers
	    voxel_offset    = voxel_offset, # x,y,z offset in voxels from the origin
	    # Pick a convenient size for your underlying chunk representation
	    # Powers of two are recommended, doesn't need to cover image exactly
	    chunk_size      = chunk_size, # units are voxels
	    volume_size     = dataset_size, # e.g. a cubic millimeter dataset
	)
	cv = CloudVolume(cloudpath, info=info)

	for mip in range(1, num_mips):
		cv.add_scale([2 ** mip, 2 ** mip, 1])

	cv.commit_info()

@main.command()
@click.argument("source")
@click.argument("destination")
def upload(source, destination):
	"""
	Process a subtile directory and upload to
	cloud storage.
	"""
	vol = CloudVolume(destination)

	for entry in tqdm(os.scandir(source)):
		if not entry.is_file():
			continue
		filename = entry.name
		ext = os.path.splitext(filename)[1]
		if ext != ".bmp":
			continue

		img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
		while img.ndim < 4:
			img = img[..., np.newaxis]

		bbx = Bbox.from_filename(get_ng(filename))
		vol[bbx] = img










