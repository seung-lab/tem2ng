from typing import List, Dict, Union, Tuple

import csv
import io
import re
import os
import multiprocessing as mp
import shutil

import click
import cv2
import imagecodecs
import numpy as np
import pathos.pools
from tqdm import tqdm
import tinybrain

from cloudvolume import CloudVolume, Bbox
from cloudvolume.exceptions import InfoUnavailableError
from cloudvolume.lib import mkdir, touch

import cloudfiles.paths
from cloudfiles import CloudFile

TILE_REGEXP = re.compile(r'tile_(\d+)_(\d+)\.bmp')

class IntTuple(click.ParamType):
  """A command line option type consisting of 3 comma-separated integers."""
  name = 'tuple'
  def convert(self, value, param, ctx):
    if isinstance(value, str):
      try:
        value = tuple(map(int, value.split(',')))
      except ValueError:
        self.fail(f"'{value}' does not contain a comma delimited list of integers.")
    return value

def decode_tilename(tilename) -> List[int]:
    return [ int(_) for _ in re.search(TILE_REGEXP, tilename).groups() ]

def read_stage_csv(source) -> List[Dict[str,Union[int,float]]]:
    """
    Example data:
    tile_id, stage_x_nm, stage_y_nm, x_relroi_nm, y_relroi_nm
    0, 338700.639, 109345.286, 329885.172, 109961.724
    1, 283719.777, 109345.286, 274904.310, 109961.724
    """
    source = cloudfiles.paths.normalize(source)
    cf = CloudFile(source)
    filepath = cf.join(source, "metadata/stage_positions.csv")
    cf = CloudFile(filepath)
    stage_data = io.StringIO(cf.get().decode("utf8"))

    stage_csv = []
    for row in csv.DictReader(stage_data):
        tile_id = int(row["tile_id"])
        row = { 
            k.strip():float(v.strip())
            for k,v in row.items() 
        }
        row["tile_id"] = tile_id
        stage_csv.append(row)
    
    return stage_csv

def compute_supertile_map(stage_csv:List[Dict[str,Union[int,float]]]) -> np.ndarray:
    # Determine the dimensions of the 2D array
    rank_x = np.unique([ row["stage_x_nm"] for row in stage_csv ])
    rank_y = np.unique([ row["stage_y_nm"] for row in stage_csv ])

    rank_x = { v:i for i,v in enumerate(rank_x) }
    rank_y = { v:i for i,v in enumerate(rank_y) }    
    
    arr = np.full((len(rank_y)+1, len(rank_x)+1), None)

    for row in stage_csv:
        arr[rank_y[row["stage_y_nm"]], rank_x[row["stage_x_nm"]]] = row["tile_id"]
    
    # Reverse the order of the rows in the array
    supertile_map = arr[::-1]
    
    return supertile_map

def compute_tile_id_map(
    stage_csv:List[Dict[str,Union[int,float]]]
) -> Dict[str, Tuple[int,int]]:

    supertile_map = compute_supertile_map(stage_csv)

    # Cricket subtile order
    # SUBTILE_MAP = [
    #     [6, 7, 8], 
    #     [5, 0, 1], 
    #     [4, 3, 2]
    # ]

    xmap = { 
        6:0, 7:1, 8:2,
        5:0, 0:1, 1:2,
        4:0, 3:1, 2:2,
    }

    ymap = {
        6:0, 7:0, 8:0,
        5:1, 0:1, 1:1,
        4:2, 3:2, 2:2,
    }

    tile_id_map = {}
    for i, supertile_row in enumerate(supertile_map):
        for j, supertile in enumerate(supertile_row):
            if supertile is None:
                continue
            for subtile in range(9):
                x = (j * 3) + xmap[subtile]
                y = ((i-1) * 3) + ymap[subtile] # first row is None
                tilename = f"tile_{supertile:04}_{subtile}"
                tile_id_map[tilename] = (x,y)

    return tile_id_map

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
@click.option('--clear-progress', is_flag=True, default=False, help="Delete the progress directory and upload from the beginning.", show_default=True)
@click.pass_context
def upload(ctx, source, destination, z, clear_progress):
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
    tile_id_map = compute_tile_id_map(stage_csv)

    done_files = set(os.listdir(mkdir(progress_dir)))
    all_files = os.listdir(subtiles_dir)
    all_files = set([
        fname for fname in all_files
        if (
            os.path.isfile(os.path.join(subtiles_dir, fname))
            and (os.path.splitext(fname)[1] == ".bmp"
            or os.path.splitext(fname)[1] == ".jxl")
        )
    ])
    to_upload = list(all_files.difference(done_files))
    to_upload.sort()
    
    def process(filename):
        nonlocal tile_id_map
        if os.path.splitext(filename)[1] == ".jxl":
            with open(os.path.join(subtiles_dir, filename), "rb") as f:
                binary = f.read()
            img = imagecodecs.jpegxl_decode(binary)
        else:
            img = cv2.imread(os.path.join(subtiles_dir, filename), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"{filename} could not be opened.")
            return 0

        img = cv2.transpose(img)

        while img.ndim < 4:
            img = img[..., np.newaxis]

        # tile_id, subtile_id = decode_tilename(filename)
        (x,y) = tile_id_map[filename.split(".")[0]]

        # padding on the top and left
        x += 1
        y += 1

        # get_ng(filename, stages[0]-west_most, stages[1]-south_most, z=z, step=step)
        bbx = Bbox((x,y,z), (x+1,y+1,z+1), dtype=int)
        bbx.minpt.x *= img.shape[0]
        bbx.maxpt.x *= img.shape[0]
        bbx.minpt.y *= img.shape[1]
        bbx.maxpt.y *= img.shape[1]
        
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
        if parallel == 1:
            for filename in to_upload:
                pbar.update(process(filename))
        else:
            with pathos.pools.ProcessPool(parallel) as pool:
                for num_inserted in pool.imap(process, to_upload):
                    pbar.update(num_inserted)

@main.command()
@click.argument("source", type=CloudPath())
@click.argument("destination", type=CloudPath())
@click.option('--resolution', type=IntTuple(), default='1,1,1', help="Set resolution of image (nm).", show_default=True)
@click.pass_context
def xray(ctx, source, destination, resolution):
    """
    Process axial views of a microCT directory and upload to
    cloud storage.
    """
    source = source.replace("file://", "")

    subtiles_dir = source
    
    all_files = os.listdir(subtiles_dir)
    all_files = [
        fname for fname in all_files
        if (
            os.path.isfile(os.path.join(subtiles_dir, fname))
            and os.path.splitext(fname)[1] == ".tif"
        )
    ]
    all_files.sort()

    one_img = cv2.imread(os.path.join(subtiles_dir, all_files[0]))

    volume_size = [
        one_img.shape[1],
        one_img.shape[0],
        len(all_files),
    ]

    full_img = np.zeros(list(volume_size) + [1], dtype=np.uint8, order="F")

    for z, filename in enumerate(all_files):
        img = cv2.imread(os.path.join(subtiles_dir, filename), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"{filename} could not be opened.")
            return 0

        img = cv2.transpose(img)

        while img.ndim < 4:
            img = img[..., np.newaxis]

        full_img[:,:,z:z+1] = img


    CloudVolume.from_numpy(
        full_img, 
        vol_path=destination,
        resolution=resolution, voxel_offset=(0,0,0), 
        chunk_size=(128,128,64), layer_type='image', max_mip=0,
        encoding='jxl', compress=None, progress=True,
        encoding_level=100, encoding_effort=1,
    )




