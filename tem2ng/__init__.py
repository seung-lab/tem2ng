import csv
import re
import os
import multiprocessing as mp
import shutil

import click
import cv2
import numpy as np
import pathos.pools
from tqdm import tqdm
import tinybrain

from cloudvolume import CloudVolume, Bbox
from cloudvolume.exceptions import InfoUnavailableError
from cloudvolume.lib import mkdir, touch

import cloudfiles.paths

TILE_REGEXP = re.compile(r'tile_(\d+)_(\d+)\.bmp')

# x_step = 42320; y_step = 42309 # x, y for larger overlap
# blade2 step: step = 44395
# blade1 step: step = 42795

def get_ng(tilename, x, y, z, step):
    t1, t2 = [ int(_) for _ in re.search(TILE_REGEXP, tilename).groups() ]

    x_map = {6:0,7:1,8:2,5:0,0:1,1:2,4:0,3:1,2:2}
    get_x = lambda t2: 6000 * (round(x / step)*3 + x_map[t2])
    y_map = {6:0,5:1,4:2,7:0,0:1,3:2,8:0,1:1,2:2}
    get_y = lambda t2: 6000 * ((35 - round(y / step))*3 + y_map[t2])

    x0 = get_x(t2)
    xf = x0 + 6000
    y0 = get_y(t2)
    yf = y0 + 6000

    return f"{x0}-{xf}_{y0}-{yf}_{z}-{z+1}"

def read_stage(path):
    with open(path) as f:
        lines = f.readlines()
    return float(lines[10].split(" = ")[1]), float(lines[11].split(" = ")[1])

def read_stage_csv(source):
    filepath = os.path.join(source, "metadata/stage_positions.csv")
    stage_csv = []
    with open(filepath) as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for row in reader:
            stage_csv.append([float(row[1]),float(row[2])])
    return stage_csv

class CloudPath(click.ParamType):
  name = "CloudPath"
  def convert(self, value, param, ctx):
    return cloudfiles.paths.normalize(value)

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
@click.option('--voxel-offset', type=Tuple3(), default="0,0,0", help="Dimensions of the dataset in voxels.", show_default=True)
@click.option('--chunk-size', type=Tuple3(), default="3000,3000,1", help="Chunk size of new layers.", show_default=True)
@click.option('--resolution', type=Tuple3(), default="1,1,1", help="Resolution of a layer in nanometers.", show_default=True)
@click.option('--bit-depth', type=int, default=8, help="Number of bits per a pixel.", show_default=True)
@click.option('--num-mips', type=int, default=3, help="Number of mip levels to generate at once.", show_default=True)
@click.option('--encoding', type=str, default="raw", help="What image encoding should be used? Options: raw, png, jxl", show_default=True)
@click.option('--jxl-effort', type=int, default=1, help="For jxl, how much effort should be spent to find the best encoding? e=1-10", show_default=True)
@click.option('--jxl-quality', type=int, default=100, help="For jxl, what quality level to use? Range: 0-100. 100 = mathematically lossless, 90 = visually lossless", show_default=True)
@click.argument("cloudpath", type=CloudPath())
def info(
    cloudpath,
    dataset_size, voxel_offset, chunk_size,
    resolution, bit_depth, num_mips,
    encoding, jxl_quality, jxl_effort,
):
    """
    Creates and uploads the neuroglancer info file.
    This defines the size and properties of the image.
    We assume a grayscale image for microscopes.
    """
    if bit_depth not in (8,16,32,64):
        print("tem2ng: bit depth must be 8, 16, 32, or 64.")
        return

    if encoding not in ["raw", "png", "jxl"]:
        print(f"tem2ng: encoding {encoding} is not valid.")

    encoding_level = None if encoding != "jxl" else jxl_quality
    encoding_effort = None if encoding != "jxl" else jxl_effort

    info = CloudVolume.create_new_info(
        num_channels    = 1,
        layer_type      = 'image',
        data_type       = f'uint{bit_depth}',
        encoding        = encoding,
        encoding_level  = encoding_level,
        encoding_effort = encoding_effort,
        resolution      = resolution, # Voxel scaling, units are in nanometers
        voxel_offset    = voxel_offset, # x,y,z offset in voxels from the origin
        # Pick a convenient size for your underlying chunk representation
        # Powers of two are recommended, doesn't need to cover image exactly
        chunk_size      = chunk_size, # units are voxels
        volume_size     = dataset_size, # e.g. a cubic millimeter dataset
    )
    cv = CloudVolume(cloudpath, info=info)

    chunk_size = np.asarray(chunk_size)

    for mip in range(1, num_mips):
        new_chunk_size = chunk_size // (2 ** mip)
        new_chunk_size[2] = 1
        new_chunk_size = new_chunk_size.astype(int)
        cv.add_scale([2 ** mip, 2 ** mip, 1], chunk_size=new_chunk_size)

    cv.commit_info()

@main.command()
@click.argument("source", type=CloudPath())
@click.argument("destination", type=CloudPath())
@click.option('--z', type=int, default=0, help="Z coordinate to upload this section to.", show_default=True)
@click.option('--step', type=int, default=BLADE2_STEP, help=f"Stage step size; Blade1 {BLADE1_STEP}; Blade2 {BLADE2_STEP}", show_default=True)
@click.option('--clear-progress', is_flag=True, default=False, help="Delete the progress directory and upload from the beginning.", show_default=True)
@click.pass_context
def upload(ctx, source, destination, z, step, clear_progress):
    """
    Process a subtile directory and upload to
    cloud storage.
    """
    vol = CloudVolume(destination)

    source = source.replace("file://", "")

    progress_dir = mkdir(os.path.join(source, '.tem2ng', 'progress'))

    if clear_progress:
        shutil.rmtree(progress_dir)

    subtiles_dir = os.path.join(source, 'subtiles')
    
    stage_csv = read_stage_csv(source)
    south_most = min([i[1] for i in stage_csv]) - (step * 4)
    west_most = min([i[0] for i in stage_csv]) - (step * 4)

    done_files = set(os.listdir(progress_dir))
    all_files = os.listdir(subtiles_dir)
    all_files = set([
        fname for fname in all_files
        if (
            os.path.isfile(os.path.join(subtiles_dir, fname))
            and os.path.splitext(fname)[1] == ".bmp"
        )
    ])
    to_upload = list(all_files.difference(done_files))
    to_upload.sort()

    def process(filename):
        img = cv2.imread(os.path.join(subtiles_dir, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.transpose(img)
        while img.ndim < 4:
            img = img[..., np.newaxis]

        stages = stage_csv[int(filename.split("_")[1])]

        bbx = Bbox.from_filename(get_ng(filename, stages[0]-west_most, stages[1]-south_most, z=z, step=step))
        
        num_mips = len(vol.meta.available_mips)
        vol[bbx] = img

        if num_mips > 1:
            mips = tinybrain.downsample_with_averaging(img, factor=(2,2,1), num_mips=(num_mips - 1), sparse=False)
            minpt = bbx.minpt.clone()
            for mip_i, mip_img in enumerate(mips):
                minpt //= 2
                minpt[2] = bbx.minpt[2]
                vol.image.upload(mip_img, minpt, mip_i + 1)

        touch(os.path.join(progress_dir, filename))
        return 1

    parallel = int(ctx.obj.get("parallel", 1))

    with tqdm(desc="Upload", total=len(all_files), initial=len(done_files)) as pbar:
        with pathos.pools.ProcessPool(parallel) as pool:
            for num_inserted in pool.imap(process, to_upload):
                pbar.update(num_inserted)
