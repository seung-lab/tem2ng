import re
import os
import multiprocessing as mp

import click
import cv2
import numpy as np
import pathos.pools
from tqdm import tqdm
import tinybrain

from cloudvolume import CloudVolume, Bbox
from cloudvolume.exceptions import InfoUnavailableError
from cloudvolume.lib import mkdir, touch

TILE_REGEXP = re.compile(r'tile_(\d+)_(\d+)\.bmp')

def get_ng(tilename, left_most, bottom_most, stage_x, stage_y, z=0):
    # stage_x, stage_y: stage coordinates in pixel value
    # left_most, bottom_most: boundary of the montage at left, bottom
    t1, t2 = [ int(_) for _ in re.search(TILE_REGEXP, tilename).groups() ]

    x_map = {6:-5400,5:-5400,4:-5400,7:0,0:0,3:0,8:5400,1:5400,2:5400}

    y_map = {6:5400,7:5400,8:5400,5:0,0:0,1:0,4:-5400,3:-5400,2:-5400}

    x0 = int(stage_x - left_most + x_map[t2])
    xf = x0 + 6000
    y0 = int(stage_y - bottom_most + y_map[t2])
    yf = y0 + 6000

    return f"{x0}-{xf}_{y0}-{yf}_{z}-{z+1}"

def read_stage(path):
    with open(path) as f:
        lines = f.readlines()
    return float(lines[10].split(" = ")[1]), float(lines[11].split(" = ")[1])

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
@click.option("-p", "--parallel", default=1, help="Run with this number of parallel processes. If 0, use number of cores.")
@click.pass_context
def main(ctx, parallel):
    """
    Upload tiles from a specific setup at PNI to
    cloud storage.
    """
    parallel = int(parallel)
    if parallel == 0:
        parallel = mp.cpu_count()
    ctx.ensure_object(dict)
    ctx.obj["parallel"] = max(min(parallel, mp.cpu_count()), 1)

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
    cv = CloudVolume(cloudpath, info=info, non_aligned_writes=True)

    for mip in range(1, num_mips):
        cv.add_scale([2 ** mip, 2 ** mip, 1])

    cv.commit_info()

@main.command()
@click.argument("source")
@click.argument("destination")
@click.pass_context
def upload(ctx, source, destination):
    """
    Process a subtile directory and upload to
    cloud storage.
    """
    vol = CloudVolume(destination)
    progress_dir = mkdir(os.path.join(source, 'progress'))


    x, y = read_stage(os.path.join(source, "tile_0_4.txt"))

    done_files = set(os.listdir(progress_dir))
    all_files = os.listdir(source)
    all_files = set([
        fname for fname in all_files
        if (
            os.path.isfile(os.path.join(source, fname))
            and os.path.splitext(fname)[1] == ".bmp"
        )
    ])
    to_upload = list(all_files.difference(done_files))
    to_upload.sort()
    res = 3

    def process(filename, left_most=x/res - 10000, bottom_most=y/res - 140000, res=res):
        img = cv2.imread(os.path.join(source, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.transpose(img)
        while img.ndim < 4:
            img = img[..., np.newaxis]

        stage_x, stage_y = read_stage(os.path.join(source, filename.replace(".bmp",".txt")))

        bbx = Bbox.from_filename(get_ng(filename, left_most, bottom_most, stage_x/res, stage_y/res))
        vol[bbx] = img
        touch(os.path.join(progress_dir, filename))
        return 1

    parallel = int(ctx.obj.get("parallel", 1))

    with tqdm(desc="Upload", total=len(all_files), initial=len(done_files)) as pbar:
        with pathos.pools.ProcessPool(parallel) as pool:
            for num_inserted in pool.imap(process, to_upload):
                pbar.update(num_inserted)
