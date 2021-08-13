import re
import os

import click

from cloudvolume import CloudVolume
import numpy as np
import tinybrain

from PIL import Image

TILE_REGEXP = re.compile(r'tile_(\d+)_(\d+)\.bmp')

def get_ng(tilename, z=0):
	x_map = {4:0,5:1,6:2,3:0,0:1,7:2,2:0,1:1,8:2}
	get_x = lambda t1,t2: 6000 * ((t1%25)*3 + x_map[t2])
	y_map = {4:0,3:1,2:2,5:0,0:1,1:2,6:0,7:1,8:2}
	get_y = lambda t1,t2: 6000 * ((t1//25)*3 + y_map[t2])

	t1, t2 = re.match(TILE_REGEXP, tilename).groups()
	
	x0 = get_x(t1, t2)
	xf = x0 + 6000
	y0 = get_y(t1,t2)
	yf = y0 + 6000

	return f"{x0}-{xf}_{y0}-{yf}_{z}-{z+1}"

@click.command()
@click.argument("source")
def main(source):
	cv = CloudVolume("matrix://pni-tem1/test/")

	for filename in os.scandir(source):
		ext = os.path.splitext(filename)[1]
		if ext != ".bmp":
			continue

		img = Image.open(filename)


	img = Image.open("tile_99_0.bmp")
	img.show()










