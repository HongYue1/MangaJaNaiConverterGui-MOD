import argparse
import ctypes
import io
import json
import os
import platform
import sys
import time
import traceback
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from queue import Full, Queue
from threading import Event, Lock, Thread
from typing import Any, Literal
from zipfile import ZipFile, ZIP_DEFLATED

import cv2
import numpy as np
import pyvips
import rarfile
from chainner_ext import ResizeFilter, resize
from cv2.typing import MatLike
from PIL import Image, ImageCms, ImageFilter
from PIL.Image import Image as ImageType
from PIL.ImageCms import ImageCmsProfile
from rarfile import RarFile
from spandrel import ImageModelDescriptor, ModelDescriptor

sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__))))

import spandrel_custom
from nodes.impl.image_utils import normalize, to_uint8, to_uint16
from nodes.impl.upscale.auto_split_tiles import (
    ESTIMATE,
    MAX_TILE_SIZE,
    NO_TILING,
    TileSize,
)
from nodes.utils.utils import get_h_w_c
from packages.chaiNNer_pytorch.pytorch.io.load_model import load_model_node
from packages.chaiNNer_pytorch.pytorch.processing.upscale_image import (
    upscale_image_node,
)
from progress_controller import ProgressController, ProgressToken

from api import (
    NodeContext,
    SettingsParser,
)


class _ExecutorNodeContext(NodeContext):
    def __init__(
        self, progress: ProgressToken, settings: SettingsParser, storage_dir: Path
    ) -> None:
        super().__init__()

        self.progress = progress
        self.__settings = settings
        self._storage_dir = storage_dir

        self.chain_cleanup_fns: set[Callable[[], None]] = set()
        self.node_cleanup_fns: set[Callable[[], None]] = set()

    @property
    def aborted(self) -> bool:
        return self.progress.aborted

    @property
    def paused(self) -> bool:
        time.sleep(0.001)
        return self.progress.paused

    def set_progress(self, progress: float) -> None:
        self.check_aborted()

        # TODO: send progress event

    @property
    def settings(self) -> SettingsParser:
        """
        Returns the settings of the current node execution.
        """
        return self.__settings

    @property
    def storage_dir(self) -> Path:
        return self._storage_dir

    def add_cleanup(
        self, fn: Callable[[], None], after: Literal["node", "chain"] = "chain"
    ) -> None:
        if after == "chain":
            self.chain_cleanup_fns.add(fn)
        elif after == "node":
            self.node_cleanup_fns.add(fn)
        else:
            raise ValueError(f"Unknown cleanup type: {after}")


def get_tile_size(tile_size_str: str) -> TileSize:
    if tile_size_str == "Auto (Estimate)":
        return ESTIMATE
    elif tile_size_str == "Maximum":
        return MAX_TILE_SIZE
    elif tile_size_str == "No Tiling":
        return NO_TILING
    elif tile_size_str.isdecimal():
        return TileSize(int(tile_size_str))

    return ESTIMATE


"""
lanczos downscale without color conversion, for pre-upscale
downscale and final color downscale
"""


def standard_resize(image: np.ndarray, new_size: tuple[int, int]) -> np.ndarray:
    new_image = image.astype(np.float32) / 255.0
    new_image = resize(new_image, new_size, ResizeFilter.Lanczos, False)
    new_image = (new_image * 255).round().astype(np.uint8)

    _, _, c = get_h_w_c(image)

    if c == 1 and new_image.ndim == 3:
        new_image = np.squeeze(new_image, axis=-1)

    return new_image


"""
final downscale for grayscale images only
"""


def dotgain20_resize(image: np.ndarray, new_size: tuple[int, int]) -> np.ndarray:
    h, _, c = get_h_w_c(image)
    size_ratio = h / new_size[1]
    blur_size = (1 / size_ratio - 1) / 3.5
    if blur_size >= 0.1:
        blur_size = min(blur_size, 250)

    pil_image = Image.fromarray(image, mode="L")
    pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_size))
    pil_image = ImageCms.applyTransform(pil_image, dotgain20togamma1transform, False)

    new_image = np.array(pil_image)
    new_image = new_image.astype(np.float32) / 255.0
    new_image = resize(new_image, new_size, ResizeFilter.CubicCatrom, False)
    new_image = (new_image * 255).round().astype(np.uint8)

    pil_image = Image.fromarray(new_image[:, :, 0], mode="L")
    pil_image = ImageCms.applyTransform(pil_image, gamma1todotgain20transform, False)
    return np.array(pil_image)


def image_resize(
    image: np.ndarray, new_size: tuple[int, int], is_grayscale: bool
) -> np.ndarray:
    if is_grayscale:
        return dotgain20_resize(image, new_size)

    return standard_resize(image, new_size)


def get_system_codepage() -> Any:
    return None if not is_windows else ctypes.windll.kernel32.GetConsoleOutputCP()


def enhance_contrast(image: np.ndarray) -> MatLike:
    image_p = Image.fromarray(image).convert("L")

    # Calculate the histogram
    hist = image_p.histogram()
    # print(hist)

    # Find the global maximum peak in the range 0-30 for the black level
    new_black_level = 0
    global_max_black = hist[0]

    for i in range(1, 31):
        if hist[i] > global_max_black:
            global_max_black = hist[i]
            new_black_level = i
        # elif hist[i] < global_max_black:
        #     break

    # Continue searching at 31 and later for the black level
    continuous_count = 0
    for i in range(31, 256):
        if hist[i] > global_max_black:
            continuous_count = 0
            global_max_black = hist[i]
            new_black_level = i
        elif hist[i] < global_max_black:
            continuous_count += 1
            if continuous_count > 1:
                break

    # Find the global maximum peak in the range 255-225 for the white level
    new_white_level = 255
    global_max_white = hist[255]

    for i in range(254, 224, -1):
        if hist[i] > global_max_white:
            global_max_white = hist[i]
            new_white_level = i
        # elif hist[i] < global_max_white:
        #     break

    # Continue searching at 224 and below for the white level
    continuous_count = 0
    for i in range(223, -1, -1):
        if hist[i] > global_max_white:
            continuous_count = 0
            global_max_white = hist[i]
            new_white_level = i
        elif hist[i] < global_max_white:
            continuous_count += 1
            if continuous_count > 1:
                break

    print(
        f"Auto adjusted levels: new black level = {new_black_level}; new white level = {new_white_level}",
        flush=True,
    )

    image_array = np.array(image_p).astype("float32")
    image_array = np.maximum(image_array - new_black_level, 0) / (
        new_white_level - new_black_level
    )
    return np.clip(image_array, 0, 1)


def _read_image(img_stream: bytes, filename: str) -> np.ndarray:
    return _read_vips(img_stream)


def _read_image_from_path(path: str) -> np.ndarray:
    return pyvips.Image.new_from_file(path, access="sequential", fail=True).icc_transform("srgb").numpy()


def _read_vips(img_stream: bytes) -> np.ndarray:
    return pyvips.Image.new_from_buffer(img_stream, "", access="sequential").icc_transform("srgb").numpy()


def cv_image_is_grayscale(image: np.ndarray, user_threshold: float) -> bool:
    _, _, c = get_h_w_c(image)

    if c == 1:
        return True

    b, g, r = cv2.split(image[:, :, :3])

    ignore_threshold = user_threshold

    # getting differences between (b,g), (r,g), (b,r) channel pixels
    r_g = cv2.subtract(cv2.absdiff(r, g), ignore_threshold)  # type: ignore
    r_b = cv2.subtract(cv2.absdiff(r, b), ignore_threshold)  # type: ignore
    g_b = cv2.subtract(cv2.absdiff(g, b), ignore_threshold)  # type: ignore

    # create masks to identify pure black and pure white pixels
    pure_black_mask = np.logical_and.reduce((r == 0, g == 0, b == 0))
    pure_white_mask = np.logical_and.reduce((r == 255, g == 255, b == 255))

    # combine masks to exclude both pure black and pure white pixels
    exclude_mask = np.logical_or(pure_black_mask, pure_white_mask)

    # exclude pure black and pure white pixels from diff_sum and image size calculation
    diff_sum = np.sum(np.where(exclude_mask, 0, r_g + r_b + g_b))
    size_without_black_and_white = np.sum(~exclude_mask) * 3

    # if the entire image is pure black or pure white, return False
    if size_without_black_and_white == 0:
        return False

    # finding ratio of diff_sum with respect to size of image without pure black and pure white pixels
    ratio = diff_sum / size_without_black_and_white

    return ratio <= user_threshold / 12


def convert_image_to_grayscale(image: np.ndarray) -> np.ndarray:
    channels = get_h_w_c(image)[2]
    if channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    return image


def get_chain_for_image(
    image: np.ndarray,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    grayscale_detection_threshold: int,
) -> tuple[dict[str, Any], bool, int, int] | tuple[None, None, int, int]:
    original_height, original_width, _ = get_h_w_c(image)

    if target_width != 0 and target_height != 0:
        target_scale = min(
            target_height / original_height, target_width / original_width
        )
    if target_height != 0:
        target_scale = target_height / original_height
    elif target_width != 0:
        target_scale = target_width / original_width

    assert target_scale is not None

    is_grayscale = cv_image_is_grayscale(image, grayscale_detection_threshold)

    for chain in chains:
        if should_chain_activate_for_image(
            original_width, original_height, is_grayscale, target_scale, chain
        ):
            print("Matched Chain:", chain, flush=True)
            return chain, is_grayscale, original_width, original_height

    return None, None, original_width, original_height


def should_chain_activate_for_image(
    original_width: int,
    original_height: int,
    is_grayscale: bool,
    target_scale: float,
    chain: dict[str, Any],
) -> bool:
    min_width, min_height = (int(x) for x in chain["MinResolution"].split("x"))
    max_width, max_height = (int(x) for x in chain["MaxResolution"].split("x"))

    # resolution tests
    if min_width != 0 and min_width > original_width:
        return False
    if min_height != 0 and min_height > original_height:
        return False
    if max_width != 0 and max_width < original_width:
        return False
    if max_height != 0 and max_height < original_height:
        return False

    # color / grayscale tests
    if is_grayscale and not chain["IsGrayscale"]:
        return False
    if not is_grayscale and not chain["IsColor"]:
        return False

    # scale tests
    if chain["MaxScaleFactor"] != 0 and target_scale > chain["MaxScaleFactor"]:
        return False
    if chain["MinScaleFactor"] != 0 and target_scale < chain["MinScaleFactor"]:
        return False

    return True


def ai_upscale_image(
    image: np.ndarray, model_tile_size: TileSize, model: ImageModelDescriptor | None
) -> np.ndarray:
    if model is not None:
        result = upscale_image_node(
            context,
            image,
            model,
            False,
            0,
            model_tile_size,
            256,
            False,
        )

        _, _, c = get_h_w_c(image)

        if c == 1 and result.ndim == 3:
            result = np.squeeze(result, axis=-1)

        return result

    return image


def postprocess_image(image: np.ndarray) -> np.ndarray:
    # print(f"postprocess_image")
    return to_uint8(image, normalized=True)


def final_target_resize(
    image: np.ndarray,
    target_scale: float,
    target_width: int,
    target_height: int,
    original_width: int,
    original_height: int,
    is_grayscale: bool,
) -> np.ndarray:
    # fit to dimensions
    if target_height != 0 and target_width != 0:
        h, w, _ = get_h_w_c(image)
        # determine whether to fit to height or width
        if target_height / original_height < target_width / original_width:
            target_width = 0
        else:
            target_height = 0

    # resize height, keep proportional width
    if target_height != 0:
        h, w, _ = get_h_w_c(image)
        if h != target_height:
            return image_resize(
                image, (round(w * target_height / h), target_height), is_grayscale
            )
    # resize width, keep proportional height
    elif target_width != 0:
        h, w, _ = get_h_w_c(image)
        if w != target_width:
            return image_resize(
                image, (target_width, round(h * target_width / w)), is_grayscale
            )
    else:
        h, w, _ = get_h_w_c(image)
        new_target_height = round(original_height * target_scale)
        if h != new_target_height:
            return image_resize(
                image,
                (round(w * new_target_height / h), new_target_height),
                is_grayscale,
            )

    return image


def get_encode_worker_count() -> int:
    """
    how many images may be encoded at the same time

    encoding happens in libvips, which releases the GIL, so these workers really
    do run in parallel with each other and with the upscale loop. override with
    the MJN_ENCODE_WORKERS environment variable.
    """
    env_worker_count = os.environ.get("MJN_ENCODE_WORKERS", "").strip()
    if env_worker_count.isdecimal() and int(env_worker_count) > 0:
        return int(env_worker_count)

    return max(2, min(4, (os.cpu_count() or 2) - 1))


def get_pipeline_queue_depth(encode_worker_count: int) -> int:
    """
    how many images may wait in each pipeline queue

    with a depth of 1, the upscale thread blocks on put() as soon as a single
    result is waiting to be encoded, so one slow encode stalls the GPU. override
    with the MJN_PIPELINE_DEPTH environment variable.
    """
    env_queue_depth = os.environ.get("MJN_PIPELINE_DEPTH", "").strip()
    if env_queue_depth.isdecimal() and int(env_queue_depth) > 0:
        return int(env_queue_depth)

    return max(2, encode_worker_count)


def encode_image_to_buffer(
    image: np.ndarray,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    original_width: int,
    original_height: int,
    target_scale: float,
    target_width: int,
    target_height: int,
    is_grayscale: bool,
) -> bytes:
    """
    resize an upscaled image to its final size and encode it into file bytes

    this is the expensive half of the old postprocess step. it runs in the encode
    workers, so several images can be encoded at the same time while the next
    image is still being upscaled.
    """
    # upscale_worker already converts to uint8, this is a no-op for uint8 input
    if image.dtype != np.uint8:
        image = to_uint8(image, normalized=True)

    image = final_target_resize(
        image,
        target_scale,
        target_width,
        target_height,
        original_width,
        original_height,
        is_grayscale,
    )

    # Convert the resized image back to bytes
    args = {"Q": int(lossy_compression_quality)}
    if image_format in {"webp"}:
        args["lossless"] = use_lossless_compression
    buf_img = pyvips.Image.new_from_array(image).write_to_buffer(f".{image_format}", **args)

    return io.BytesIO(buf_img).getvalue()  # type: ignore


def write_image_file(output_file_path: str, image_data: bytes) -> None:
    """
    write already encoded image bytes to disk
    """
    print(f"save image: {output_file_path}", flush=True)

    output_file_directory = os.path.dirname(output_file_path)
    if output_file_directory:
        os.makedirs(output_file_directory, exist_ok=True)

    with open(output_file_path, "wb") as output_file:
        output_file.write(image_data)


PIPELINE_PUT_POLL_SECONDS = 0.5


class PipelineConsumerGone(RuntimeError):
    """
    raised by PipelineQueue.put when no consumer is left to take the item
    """


class PipelineQueue(Queue):
    """
    a bounded queue whose put() stops waiting once its consumers are gone

    Queue.put() on a full queue waits forever, so a consumer that dies mid-job
    leaves its producer blocked on a queue nobody will ever drain again: the
    job hangs with no error, which is the missing-sentinel hang in reverse.
    Every consumer reports its own exit here, and put() rechecks that while it
    waits, so a producer is told to stop instead of waiting for a reader that
    no longer exists.
    """

    def __init__(self, maxsize: int = 0, consumer_count: int = 1) -> None:
        super().__init__(maxsize=maxsize)
        self._live_consumer_count = consumer_count
        self._live_consumer_lock = Lock()
        self._consumers_gone = Event()

    def consumer_exited(self) -> None:
        """
        report that one consumer of this queue has stopped reading

        the flag is only set when the last consumer leaves, because the
        remaining ones still drain the queue.
        """
        with self._live_consumer_lock:
            self._live_consumer_count -= 1
            if self._live_consumer_count <= 0:
                self._consumers_gone.set()

    def put(
        self, item: Any, block: bool = True, timeout: float | None = None
    ) -> None:
        """
        put an item on the queue, waiting only while a consumer could take it
        """
        if not block or timeout is not None:
            super().put(item, block=block, timeout=timeout)
            return

        while not self._consumers_gone.is_set():
            try:
                super().put(item, timeout=PIPELINE_PUT_POLL_SECONDS)
                return
            except Full:
                continue

        raise PipelineConsumerGone("every consumer of this queue has exited")

    def put_sentinel(self, sentinel: Any) -> bool:
        """
        put a sentinel, accepting that the consumers may already be gone

        a sentinel exists only to release a consumer, so there is nothing left
        to report when none of them is waiting for it. sentinels are emitted
        from finally blocks, which must not raise over the exception that
        brought the worker down in the first place.
        """
        try:
            self.put(sentinel)
        except PipelineConsumerGone:
            return False

        return True


def run_preprocess_worker(
    upscale_queue: PipelineQueue,
    preprocess_worker: Callable[..., None],
    *args: Any,
) -> None:
    """
    run a preprocess worker and always release the upscale worker afterwards

    upscale_worker only stops when it reads UPSCALE_SENTINEL, so whoever fills
    the upscale queue has to emit it however it leaves: normally, by raising,
    or by returning early (for example "file exists, skip"). Emitting it from a
    finally block here is what keeps a dead preprocess thread from leaving the
    upscale thread blocked on get() forever, which used to hang the whole job.
    """
    try:
        preprocess_worker(upscale_queue, *args)
    except PipelineConsumerGone:
        # the upscale worker already left and printed why, so the images that
        # are left have nowhere to go
        print("upscale worker exited, no more images will be queued", flush=True)
    except Exception as e:
        print(
            f"preprocess failed, no more images will be queued: {e}",
            flush=True,
        )
        traceback.print_exc()
    finally:
        upscale_queue.put_sentinel(UPSCALE_SENTINEL)


def preprocess_worker_archive(
    upscale_queue: PipelineQueue,
    input_archive_path: str,
    output_archive_path: str,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
) -> None:
    """
    given a zip or rar path, read images out of the archive, apply auto levels, add the image to upscale queue
    """

    if input_archive_path.endswith(ZIP_EXTENSIONS):
        with ZipFile(input_archive_path, "r") as input_zip:
            preprocess_worker_archive_file(
                upscale_queue,
                input_zip,
                output_archive_path,
                target_scale,
                target_width,
                target_height,
                chains,
                loaded_models,
                grayscale_detection_threshold,
            )
    elif input_archive_path.endswith(RAR_EXTENSIONS):
        with rarfile.RarFile(input_archive_path, "r") as input_rar:
            preprocess_worker_archive_file(
                upscale_queue,
                input_rar,
                output_archive_path,
                target_scale,
                target_width,
                target_height,
                chains,
                loaded_models,
                grayscale_detection_threshold,
            )


def preprocess_worker_archive_file(
    upscale_queue: PipelineQueue,
    input_archive: RarFile | ZipFile,
    output_archive_path: str,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
) -> None:
    """
    given an input zip or rar archive, read images out of the archive, apply auto levels, add the image to upscale queue
    """
    os.makedirs(os.path.dirname(output_archive_path), exist_ok=True)
    namelist = input_archive.namelist()
    print(f"TOTALZIP={len(namelist)}", flush=True)
    for filename in namelist:
        decoded_filename = filename
        image_data = None
        try:
            decoded_filename = decoded_filename.encode("cp437").decode(
                f"cp{system_codepage}"
            )
        except:  # noqa: E722
            pass

        # Open the file inside the input zip
        try:
            with input_archive.open(filename) as file_in_archive:
                # Read the image data

                image_data = file_in_archive.read()

                # image_bytes = io.BytesIO(image_data)
                image = _read_image(image_data, filename)
                print("read image", filename, flush=True)
                chain, is_grayscale, original_width, original_height = (
                    get_chain_for_image(
                        image,
                        target_scale,
                        target_width,
                        target_height,
                        chains,
                        grayscale_detection_threshold,
                    )
                )

                if is_grayscale:
                    image = convert_image_to_grayscale(image)

                model = None
                tile_size_str = ""
                if chain is not None:
                    resize_width_before_upscale = chain["ResizeWidthBeforeUpscale"]
                    resize_height_before_upscale = chain["ResizeHeightBeforeUpscale"]
                    resize_factor_before_upscale = chain["ResizeFactorBeforeUpscale"]

                    # resize width and height, distorting image
                    if (
                        resize_height_before_upscale != 0
                        and resize_width_before_upscale != 0
                    ):
                        h, w, _ = get_h_w_c(image)
                        image = standard_resize(
                            image,
                            (resize_width_before_upscale, resize_height_before_upscale),
                        )
                    # resize height, keep proportional width
                    elif resize_height_before_upscale != 0:
                        h, w, _ = get_h_w_c(image)
                        image = standard_resize(
                            image,
                            (
                                round(w * resize_height_before_upscale / h),
                                resize_height_before_upscale,
                            ),
                        )
                    # resize width, keep proportional height
                    elif resize_width_before_upscale != 0:
                        h, w, _ = get_h_w_c(image)
                        image = standard_resize(
                            image,
                            (
                                resize_width_before_upscale,
                                round(h * resize_width_before_upscale / w),
                            ),
                        )
                    elif resize_factor_before_upscale != 100:
                        h, w, _ = get_h_w_c(image)
                        image = standard_resize(
                            image,
                            (
                                round(w * resize_factor_before_upscale / 100),
                                round(h * resize_factor_before_upscale / 100),
                            ),
                        )

                    if is_grayscale and chain["AutoAdjustLevels"]:
                        image = enhance_contrast(image)
                    else:
                        image = normalize(image)

                    model_abs_path = get_model_abs_path(chain["ModelFilePath"])

                    if model_abs_path in loaded_models:
                        model = loaded_models[model_abs_path]

                    elif os.path.exists(model_abs_path):
                        model, _, _ = load_model_node(context, Path(model_abs_path))
                        loaded_models[model_abs_path] = model

                    tile_size_str = chain["ModelTileSize"]
                else:
                    image = normalize(image)

                # image = np.ascontiguousarray(image)
                upscale_queue.put(
                    (
                        image,
                        decoded_filename,
                        True,
                        is_grayscale,
                        original_width,
                        original_height,
                        get_tile_size(tile_size_str),
                        model,
                    )
                )
        except PipelineConsumerGone:
            # nothing is wrong with this file: there is no consumer left at all
            raise
        except Exception as e:
            print(
                f"could not read as image, copying file to zip instead of upscaling: {decoded_filename}, {e}",
                flush=True,
            )
            upscale_queue.put(
                (image_data, decoded_filename, False, False, None, None, None, None)
            )
        #     pass

    # the sentinel belongs to run_preprocess_worker, so that failing to open or
    # read the archive releases the upscale worker just the same
    # print("preprocess_worker_archive exiting")


def preprocess_worker_folder(
    upscale_queue: PipelineQueue,
    input_folder_path: str,
    output_folder_path: str,
    output_filename: str,
    upscale_images: bool,
    upscale_archives: bool,
    overwrite_existing_files: bool,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
) -> None:
    """
    given a folder path, recursively iterate the folder
    """
    print(
        f"preprocess_worker_folder entering {input_folder_path} {output_folder_path} {output_filename}",
        flush=True,
    )
    for root, _dirs, files in os.walk(input_folder_path):
        for filename in files:
            # for output file, create dirs if necessary, or skip if file exists and overwrite not enabled
            input_file_base = Path(filename).stem
            filename_rel = os.path.relpath(
                os.path.join(root, filename), input_folder_path
            )
            output_filename_rel = os.path.join(
                os.path.dirname(filename_rel),
                output_filename.replace("%filename%", input_file_base),
            )
            output_file_path = Path(
                os.path.join(output_folder_path, output_filename_rel)
            )

            if filename.lower().endswith(IMAGE_EXTENSIONS):  # TODO if image
                if upscale_images:
                    output_file_path = str(
                        Path(f"{output_file_path}.{image_format}")
                    ).replace("%filename%", input_file_base)

                    if not overwrite_existing_files and os.path.isfile(
                        output_file_path
                    ):
                        print(f"file exists, skip: {output_file_path}", flush=True)
                        continue

                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                    image = _read_image_from_path(os.path.join(root, filename))

                    chain, is_grayscale, original_width, original_height = (
                        get_chain_for_image(
                            image,
                            target_scale,
                            target_width,
                            target_height,
                            chains,
                            grayscale_detection_threshold,
                        )
                    )

                    if is_grayscale:
                        image = convert_image_to_grayscale(image)

                    model = None
                    tile_size_str = ""
                    if chain is not None:
                        resize_width_before_upscale = chain["ResizeWidthBeforeUpscale"]
                        resize_height_before_upscale = chain[
                            "ResizeHeightBeforeUpscale"
                        ]
                        resize_factor_before_upscale = chain[
                            "ResizeFactorBeforeUpscale"
                        ]

                        # resize width and height, distorting image
                        if (
                            resize_height_before_upscale != 0
                            and resize_width_before_upscale != 0
                        ):
                            h, w, _ = get_h_w_c(image)
                            image = standard_resize(
                                image,
                                (
                                    resize_width_before_upscale,
                                    resize_height_before_upscale,
                                ),
                            )
                        # resize height, keep proportional width
                        elif resize_height_before_upscale != 0:
                            h, w, _ = get_h_w_c(image)
                            image = standard_resize(
                                image,
                                (
                                    round(w * resize_height_before_upscale / h),
                                    resize_height_before_upscale,
                                ),
                            )
                        # resize width, keep proportional height
                        elif resize_width_before_upscale != 0:
                            h, w, _ = get_h_w_c(image)
                            image = standard_resize(
                                image,
                                (
                                    resize_width_before_upscale,
                                    round(h * resize_width_before_upscale / w),
                                ),
                            )
                        elif resize_factor_before_upscale != 100:
                            h, w, _ = get_h_w_c(image)
                            image = standard_resize(
                                image,
                                (
                                    round(w * resize_factor_before_upscale / 100),
                                    round(h * resize_factor_before_upscale / 100),
                                ),
                            )

                        if is_grayscale and chain["AutoAdjustLevels"]:
                            image = enhance_contrast(image)
                        else:
                            image = normalize(image)

                        model_abs_path = get_model_abs_path(chain["ModelFilePath"])

                        if model_abs_path in loaded_models:
                            model = loaded_models[model_abs_path]

                        elif os.path.exists(model_abs_path):
                            model, _, _ = load_model_node(context, Path(model_abs_path))
                            loaded_models[model_abs_path] = model
                        tile_size_str = chain["ModelTileSize"]
                    else:
                        image = normalize(image)

                    # image = np.ascontiguousarray(image)

                    upscale_queue.put(
                        (
                            image,
                            output_filename_rel,
                            True,
                            is_grayscale,
                            original_width,
                            original_height,
                            get_tile_size(tile_size_str),
                            model,
                        )
                    )
            elif filename.lower().endswith(ARCHIVE_EXTENSIONS):
                if upscale_archives:
                    output_file_path = f"{output_file_path}.cbz"
                    if not overwrite_existing_files and os.path.isfile(
                        output_file_path
                    ):
                        print(f"file exists, skip: {output_file_path}", flush=True)
                        continue
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                    upscale_archive_file(
                        os.path.join(root, filename),
                        output_file_path,
                        image_format,
                        lossy_compression_quality,
                        use_lossless_compression,
                        target_scale,
                        target_width,
                        target_height,
                        chains,
                        loaded_models,
                        grayscale_detection_threshold,
                    )  # TODO custom output extension
    # print("preprocess_worker_folder exiting")


def preprocess_worker_image(
    upscale_queue: PipelineQueue,
    input_image_path: str,
    output_image_path: str,
    overwrite_existing_files: bool,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
) -> None:
    """
    given an image path, apply auto levels and add to upscale queue
    """
    if input_image_path.lower().endswith(IMAGE_EXTENSIONS):
        if not overwrite_existing_files and os.path.isfile(output_image_path):
            print(f"file exists, skip: {output_image_path}", flush=True)
            return

        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        # with Image.open(input_image_path) as img:
        image = _read_image_from_path(input_image_path)

        chain, is_grayscale, original_width, original_height = get_chain_for_image(
            image,
            target_scale,
            target_width,
            target_height,
            chains,
            grayscale_detection_threshold,
        )

        if is_grayscale:
            image = convert_image_to_grayscale(image)

        model = None
        tile_size_str = ""
        if chain is not None:
            resize_width_before_upscale = chain["ResizeWidthBeforeUpscale"]
            resize_height_before_upscale = chain["ResizeHeightBeforeUpscale"]
            resize_factor_before_upscale = chain["ResizeFactorBeforeUpscale"]

            # resize width and height, distorting image
            if resize_height_before_upscale != 0 and resize_width_before_upscale != 0:
                h, w, _ = get_h_w_c(image)
                image = standard_resize(
                    image, (resize_width_before_upscale, resize_height_before_upscale)
                )
            # resize height, keep proportional width
            elif resize_height_before_upscale != 0:
                h, w, _ = get_h_w_c(image)
                image = standard_resize(
                    image,
                    (
                        round(w * resize_height_before_upscale / h),
                        resize_height_before_upscale,
                    ),
                )
            # resize width, keep proportional height
            elif resize_width_before_upscale != 0:
                h, w, _ = get_h_w_c(image)
                image = standard_resize(
                    image,
                    (
                        resize_width_before_upscale,
                        round(h * resize_width_before_upscale / w),
                    ),
                )
            elif resize_factor_before_upscale != 100:
                h, w, _ = get_h_w_c(image)
                image = standard_resize(
                    image,
                    (
                        round(w * resize_factor_before_upscale / 100),
                        round(h * resize_factor_before_upscale / 100),
                    ),
                )

            if is_grayscale and chain["AutoAdjustLevels"]:
                image = enhance_contrast(image)
            else:
                image = normalize(image)

            if chain["ModelFilePath"] == "No Model":
                pass
            else:
                model_abs_path = get_model_abs_path(chain["ModelFilePath"])

                if not os.path.exists(model_abs_path):
                    raise FileNotFoundError(model_abs_path)

                if model_abs_path in loaded_models:
                    model = loaded_models[model_abs_path]

                elif os.path.exists(model_abs_path):
                    model, _, _ = load_model_node(context, Path(model_abs_path))
                    loaded_models[model_abs_path] = model
                tile_size_str = chain["ModelTileSize"]
        else:
            print("No chain!!!!!!!")
            image = normalize(image)

        # image = np.ascontiguousarray(image)

        upscale_queue.put(
            (
                image,
                None,
                True,
                is_grayscale,
                original_width,
                original_height,
                get_tile_size(tile_size_str),
                model,
            )
        )


def upscale_worker(
    upscale_queue: PipelineQueue,
    encode_queue: PipelineQueue,
    encode_worker_count: int = 1,
) -> None:
    """
    wait for upscale queue, for each queue entry, upscale image and add result to encode queue

    the result is converted to uint8 before it is queued: that is the same order
    of operations as before, it makes a queued image use four times less memory,
    and it leaves the encode workers with only resizing and encoding to do.
    """
    # print("upscale_worker entering")
    sequence_number = 0
    try:
        while True:
            (
                image,
                file_name,
                is_image,
                is_grayscale,
                original_width,
                original_height,
                model_tile_size,
                model,
            ) = upscale_queue.get()
            if image is None:
                break

            if is_image:
                image = ai_upscale_image(image, model_tile_size, model)

                # convert back to grayscale
                if is_grayscale:
                    image = convert_image_to_grayscale(image)

                image = to_uint8(image, normalized=True)

            # the sequence number lets the writer restore the original order,
            # because the encode workers finish out of order
            encode_queue.put(
                (
                    sequence_number,
                    image,
                    file_name,
                    is_image,
                    is_grayscale,
                    original_width,
                    original_height,
                )
            )
            sequence_number += 1
    except PipelineConsumerGone:
        print(
            "every encode worker exited, no more images will be encoded",
            flush=True,
        )
    finally:
        # a preprocess worker blocked on a full upscale queue has to learn
        # that its only reader is gone
        upscale_queue.consumer_exited()
        # one sentinel per encode worker, in a finally block so that a failed
        # upscale cannot leave the rest of the pipeline waiting forever
        for _ in range(max(1, encode_worker_count)):
            encode_queue.put_sentinel(ENCODE_SENTINEL)
    # print("upscale_worker exiting")


def encode_worker(
    encode_queue: PipelineQueue,
    write_queue: PipelineQueue,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float,
    target_width: int,
    target_height: int,
) -> None:
    """
    wait for encode queue, encode each upscaled image into file bytes, pass the
    bytes on to the write queue

    several of these run at the same time, so an image no longer waits for the
    previous image to finish encoding, and the upscaler only waits for an
    encoder when every worker is already busy.
    """
    try:
        while True:
            (
                sequence_number,
                image,
                file_name,
                is_image,
                is_grayscale,
                original_width,
                original_height,
            ) = encode_queue.get()
            if sequence_number is None:
                break

            image_data = image
            if is_image:
                try:
                    image_data = encode_image_to_buffer(
                        image,
                        image_format,
                        lossy_compression_quality,
                        use_lossless_compression,
                        original_width,
                        original_height,
                        target_scale,
                        target_width,
                        target_height,
                        is_grayscale,
                    )
                except Exception as e:
                    print(
                        f"could not encode image, skipping: {file_name}, {e}",
                        flush=True,
                    )
                    image_data = None

            write_queue.put((sequence_number, file_name, image_data, is_image))
    except PipelineConsumerGone:
        print("writer exited, encoded images are being dropped", flush=True)
    finally:
        encode_queue.consumer_exited()
        write_queue.put_sentinel(WRITE_SENTINEL)


def start_encode_workers(
    encode_queue: PipelineQueue,
    write_queue: PipelineQueue,
    encode_worker_count: int,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float,
    target_width: int,
    target_height: int,
) -> list[Thread]:
    """
    start the encode workers that sit between the upscaler and the writer
    """
    encode_processes = [
        Thread(
            target=encode_worker,
            args=(
                encode_queue,
                write_queue,
                image_format,
                lossy_compression_quality,
                use_lossless_compression,
                target_scale,
                target_width,
                target_height,
            ),
        )
        for _ in range(encode_worker_count)
    ]
    for encode_process in encode_processes:
        encode_process.start()

    return encode_processes


def ordered_write_worker(
    write_queue: PipelineQueue,
    encode_worker_count: int,
    write_entry: Callable[[str, Any, bool], None],
) -> None:
    """
    consume the write queue and write each entry in the original image order

    an image that is encoded early is held back until every earlier image has
    been written, so parallel encoding can never reorder the pages of an archive
    or the progress output.
    """
    pending: dict[int, tuple[str, Any, bool]] = {}
    next_sequence_number = 0
    finished_encode_workers = 0

    def write_pending_entry(sequence_number: int) -> None:
        file_name, image_data, is_image = pending.pop(sequence_number)
        try:
            write_entry(file_name, image_data, is_image)
        except Exception as e:
            print(f"could not write image: {file_name}, {e}", flush=True)

    try:
        while finished_encode_workers < encode_worker_count:
            sequence_number, file_name, image_data, is_image = write_queue.get()
            if sequence_number is None:
                finished_encode_workers += 1
                continue

            pending[sequence_number] = (file_name, image_data, is_image)

            while next_sequence_number in pending:
                write_pending_entry(next_sequence_number)
                next_sequence_number += 1

        # only reachable if a sequence number went missing, and dropping images
        # silently would be worse than writing them late
        for sequence_number in sorted(pending):
            write_pending_entry(sequence_number)
    finally:
        # an encode worker blocked on a full write queue would otherwise wait
        # for a writer that is no longer there
        write_queue.consumer_exited()


def postprocess_worker_zip(
    write_queue: PipelineQueue,
    output_zip_path: str,
    image_format: str,
    encode_worker_count: int,
) -> None:
    """
    wait for write queue, for each queue entry, save the image to the zip file

    a single writer owns the zip file, and writes the already encoded bytes in
    the original page order.
    """
    # print("postprocess_worker_zip entering")
    output_zip_directory = os.path.dirname(output_zip_path)
    if output_zip_directory:
        os.makedirs(output_zip_directory, exist_ok=True)

    with ZipFile(output_zip_path, "w", ZIP_DEFLATED) as output_zip:

        def write_entry(file_name: str, image_data: Any, is_image: bool) -> None:
            if image_data is not None:
                if is_image:
                    entry_name = str(Path(file_name).with_suffix(f".{image_format}"))
                    print(f"save image to zip: {entry_name}", flush=True)
                    output_zip.writestr(entry_name, image_data)
                else:  # copy file
                    output_zip.writestr(file_name, image_data)
            print("PROGRESS=postprocess_worker_zip_image", flush=True)

        ordered_write_worker(write_queue, encode_worker_count, write_entry)

    print("PROGRESS=postprocess_worker_zip_archive", flush=True)


def postprocess_worker_folder(
    write_queue: PipelineQueue,
    output_folder_path: str,
    image_format: str,
    encode_worker_count: int,
) -> None:
    """
    wait for write queue, for each queue entry, save the image to the output folder
    """
    # print("postprocess_worker_folder entering")

    def write_entry(file_name: str, image_data: Any, _is_image: bool) -> None:
        if image_data is not None:
            write_image_file(
                os.path.join(
                    output_folder_path, str(Path(f"{file_name}.{image_format}"))
                ),
                image_data,
            )
        print("PROGRESS=postprocess_worker_folder", flush=True)

    ordered_write_worker(write_queue, encode_worker_count, write_entry)

    # print("postprocess_worker_folder exiting")


def postprocess_worker_image(
    write_queue: PipelineQueue,
    output_file_path: str,
    encode_worker_count: int,
) -> None:
    """
    wait for write queue, for each queue entry, save the image to the output file path
    """

    def write_entry(_file_name: str, image_data: Any, _is_image: bool) -> None:
        if image_data is not None:
            write_image_file(output_file_path, image_data)
        print("PROGRESS=postprocess_worker_image", flush=True)

    ordered_write_worker(write_queue, encode_worker_count, write_entry)


def upscale_archive_file(
    input_zip_path: str,
    output_zip_path: str,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
) -> None:
    # TODO accept multiple paths to reuse simple queues?

    encode_worker_count = get_encode_worker_count()
    queue_depth = get_pipeline_queue_depth(encode_worker_count)

    # bounded queues that tell a producer when their consumers are gone, so a
    # worker that dies downstream cannot leave one blocked on put() forever
    upscale_queue = PipelineQueue(maxsize=queue_depth)
    encode_queue = PipelineQueue(
        maxsize=queue_depth, consumer_count=encode_worker_count
    )
    write_queue = PipelineQueue(maxsize=queue_depth)

    # start preprocess zip process
    preprocess_process = Thread(
        target=run_preprocess_worker,
        args=(
            upscale_queue,
            preprocess_worker_archive,
            input_zip_path,
            output_zip_path,
            target_scale,
            target_width,
            target_height,
            chains,
            loaded_models,
            grayscale_detection_threshold,
        ),
    )
    preprocess_process.start()

    # start upscale process
    upscale_process = Thread(
        target=upscale_worker,
        args=(upscale_queue, encode_queue, encode_worker_count),
    )
    upscale_process.start()

    # start encode processes, which encode images in parallel so that the
    # upscaler never waits for an image to finish encoding
    encode_processes = start_encode_workers(
        encode_queue,
        write_queue,
        encode_worker_count,
        image_format,
        lossy_compression_quality,
        use_lossless_compression,
        target_scale,
        target_width,
        target_height,
    )

    # start postprocess zip process
    postprocess_process = Thread(
        target=postprocess_worker_zip,
        args=(
            write_queue,
            output_zip_path,
            image_format,
            encode_worker_count,
        ),
    )
    postprocess_process.start()

    # wait for all processes
    preprocess_process.join()
    upscale_process.join()
    for encode_process in encode_processes:
        encode_process.join()
    postprocess_process.join()


def upscale_image_file(
    input_image_path: str,
    output_image_path: str,
    overwrite_existing_files: bool,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
) -> None:
    encode_worker_count = get_encode_worker_count()
    queue_depth = get_pipeline_queue_depth(encode_worker_count)

    # bounded queues that tell a producer when their consumers are gone, so a
    # worker that dies downstream cannot leave one blocked on put() forever
    upscale_queue = PipelineQueue(maxsize=queue_depth)
    encode_queue = PipelineQueue(
        maxsize=queue_depth, consumer_count=encode_worker_count
    )
    write_queue = PipelineQueue(maxsize=queue_depth)

    # start preprocess image process
    preprocess_process = Thread(
        target=run_preprocess_worker,
        args=(
            upscale_queue,
            preprocess_worker_image,
            input_image_path,
            output_image_path,
            overwrite_existing_files,
            target_scale,
            target_width,
            target_height,
            chains,
            loaded_models,
            grayscale_detection_threshold,
        ),
    )
    preprocess_process.start()

    # start upscale process
    upscale_process = Thread(
        target=upscale_worker,
        args=(upscale_queue, encode_queue, encode_worker_count),
    )
    upscale_process.start()

    # start encode processes
    encode_processes = start_encode_workers(
        encode_queue,
        write_queue,
        encode_worker_count,
        image_format,
        lossy_compression_quality,
        use_lossless_compression,
        target_scale,
        target_width,
        target_height,
    )

    # start postprocess image process
    postprocess_process = Thread(
        target=postprocess_worker_image,
        args=(
            write_queue,
            output_image_path,
            encode_worker_count,
        ),
    )
    postprocess_process.start()

    # wait for all processes
    preprocess_process.join()
    upscale_process.join()
    for encode_process in encode_processes:
        encode_process.join()
    postprocess_process.join()


def upscale_file(
    input_file_path: str,
    output_folder_path: str,
    output_filename: str,
    overwrite_existing_files: bool,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
) -> None:
    input_file_base = Path(input_file_path).stem

    if input_file_path.lower().endswith(ARCHIVE_EXTENSIONS):
        output_file_path = str(
            Path(
                f"{os.path.join(output_folder_path,output_filename.replace('%filename%', input_file_base))}.cbz"
            )
        )
        print("output_file_path", output_file_path, flush=True)
        if not overwrite_existing_files and os.path.isfile(output_file_path):
            print(f"file exists, skip: {output_file_path}", flush=True)
            return

        upscale_archive_file(
            input_file_path,
            output_file_path,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            target_scale,
            target_width,
            target_height,
            chains,
            loaded_models,
            grayscale_detection_threshold,
        )

    elif input_file_path.lower().endswith(IMAGE_EXTENSIONS):
        output_file_path = str(
            Path(
                f"{os.path.join(output_folder_path,output_filename.replace('%filename%', input_file_base))}.{image_format}"
            )
        )
        if not overwrite_existing_files and os.path.isfile(output_file_path):
            print(f"file exists, skip: {output_file_path}", flush=True)
            return

        upscale_image_file(
            input_file_path,
            output_file_path,
            overwrite_existing_files,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            target_scale,
            target_width,
            target_height,
            chains,
            loaded_models,
            grayscale_detection_threshold,
        )


def upscale_folder(
    input_folder_path: str,
    output_folder_path: str,
    output_filename: str,
    upscale_images: bool,
    upscale_archives: bool,
    overwrite_existing_files: bool,
    image_format: str,
    lossy_compression_quality: int,
    use_lossless_compression: bool,
    target_scale: float | None,
    target_width: int,
    target_height: int,
    chains: list[dict[str, Any]],
    loaded_models: dict[str, ModelDescriptor],
    grayscale_detection_threshold: int,
) -> None:
    # print("upscale_folder: entering")

    # preprocess_queue = Queue(maxsize=1)
    encode_worker_count = get_encode_worker_count()
    queue_depth = get_pipeline_queue_depth(encode_worker_count)

    # bounded queues that tell a producer when their consumers are gone, so a
    # worker that dies downstream cannot leave one blocked on put() forever
    upscale_queue = PipelineQueue(maxsize=queue_depth)
    encode_queue = PipelineQueue(
        maxsize=queue_depth, consumer_count=encode_worker_count
    )
    write_queue = PipelineQueue(maxsize=queue_depth)

    # start preprocess folder process
    preprocess_process = Thread(
        target=run_preprocess_worker,
        args=(
            upscale_queue,
            preprocess_worker_folder,
            input_folder_path,
            output_folder_path,
            output_filename,
            upscale_images,
            upscale_archives,
            overwrite_existing_files,
            image_format,
            lossy_compression_quality,
            use_lossless_compression,
            target_scale,
            target_width,
            target_height,
            chains,
            loaded_models,
            grayscale_detection_threshold,
        ),
    )
    preprocess_process.start()

    # start upscale process
    upscale_process = Thread(
        target=upscale_worker,
        args=(upscale_queue, encode_queue, encode_worker_count),
    )
    upscale_process.start()

    # start encode processes
    encode_processes = start_encode_workers(
        encode_queue,
        write_queue,
        encode_worker_count,
        image_format,
        lossy_compression_quality,
        use_lossless_compression,
        target_scale,
        target_width,
        target_height,
    )

    # start postprocess folder process
    postprocess_process = Thread(
        target=postprocess_worker_folder,
        args=(
            write_queue,
            output_folder_path,
            image_format,
            encode_worker_count,
        ),
    )
    postprocess_process.start()

    # wait for all processes
    preprocess_process.join()
    upscale_process.join()
    for encode_process in encode_processes:
        encode_process.join()
    postprocess_process.join()


current_file_directory = os.path.dirname(os.path.abspath(__file__))


def get_model_abs_path(chain_model_file_path: str) -> str:
    return os.path.abspath(os.path.join(models_directory, chain_model_file_path))


def get_gamma_icc_profile() -> ImageCmsProfile:
    profile_path = os.path.join(
        current_file_directory, "../ImageMagick/Custom Gray Gamma 1.0.icc"
    )
    return ImageCms.getOpenProfile(profile_path)


def get_dot20_icc_profile() -> ImageCmsProfile:
    profile_path = os.path.join(
        current_file_directory, "../ImageMagick/Dot Gain 20%.icc"
    )
    return ImageCms.getOpenProfile(profile_path)


def parse_settings_from_cli():
    parser = argparse.ArgumentParser(prog="python run_upscale.py",
                                     description="By default, used by MangaJaNaiConverterGui as an internal tool. "
                                                 "Alternative options made available to make it easier to skip the GUI "
                                                 "and run upscaling jobs directly from CLI.")

    execution_type_group = parser.add_mutually_exclusive_group(required=True)
    execution_type_group.add_argument("--settings",
                                      help="Default behaviour, based on provided appstate configuration. "
                                           "For advanced usage.")
    execution_type_group.add_argument("-f", "--file-path",
                                      help="Upscale single file")
    execution_type_group.add_argument("-d", "--folder-path",
                                      help="Upscale whole directory")

    parser.add_argument("-o", "--output-folder-path",
                        default=os.path.join(".", "out"),
                        help="Output directory for upscaled files. Default: ./out")
    parser.add_argument("-m", "--models-directory-path",
                        default=os.path.join("..", "models"),
                        help="Directory with models used for upscaling. "
                             "Supports only models bundled with MangaJaNaiConvertedGui. "
                             "Default: MangaJaNaiConverterGui/chaiNNer/models/")
    parser.add_argument("-u", "--upscale-factor",
                        type=int,
                        choices=[1, 2, 3, 4],
                        default=2,
                        help="Used for calculating which model will be used. Default: 2")
    parser.add_argument("--device-index",
                        type=int,
                        default=0,
                        help="Device used to run upscaling jobs in case more than one is available. Default: 0")

    args = parser.parse_args()

    return parse_auto_settings(args) if args.settings else parse_manual_settings(args)


def parse_auto_settings(args):
    with open(args.settings, encoding="utf-8") as f:
        json_settings = json.load(f)

    return json_settings


def parse_manual_settings(args):
    default_file_path = os.path.join("..", "resources", "default_cli_configuration.json")
    with open(default_file_path, "r") as default_file:
        default_json = json.load(default_file)

    default_json["SelectedDeviceIndex"] = int(args.device_index)
    default_json["ModelsDirectory"] = args.models_directory_path

    default_json["Workflows"]["$values"][0]["OutputFolderPath"] = args.output_folder_path
    default_json["Workflows"]["$values"][0]["SelectedDeviceIndex"] = args.device_index
    default_json["Workflows"]["$values"][0]["UpscaleScaleFactor"] = args.upscale_factor
    if args.file_path:
        default_json["Workflows"]["$values"][0]["SelectedTabIndex"] = 0
        default_json["Workflows"]["$values"][0]["InputFilePath"] = args.file_path
    elif args.folder_path:
        default_json["Workflows"]["$values"][0]["SelectedTabIndex"] = 1
        default_json["Workflows"]["$values"][0]["InputFolderPath"] = args.folder_path

    return default_json


is_windows = platform.system() == "win32"
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

settings = parse_settings_from_cli()

workflow = settings["Workflows"]["$values"][settings["SelectedWorkflowIndex"]]
models_directory = settings["ModelsDirectory"]

UPSCALE_SENTINEL = (None, None, None, None, None, None, None, None)
ENCODE_SENTINEL = (None, None, None, None, None, None, None)
WRITE_SENTINEL = (None, None, None, None)
CV2_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
IMAGE_EXTENSIONS = (*CV2_IMAGE_EXTENSIONS, ".avif")
ZIP_EXTENSIONS = (".zip", ".cbz")
RAR_EXTENSIONS = (".rar", ".cbr")
ARCHIVE_EXTENSIONS = ZIP_EXTENSIONS + RAR_EXTENSIONS
loaded_models = {}
system_codepage = get_system_codepage()

settings_parser = SettingsParser(
    {
        "use_cpu": settings["SelectedDeviceIndex"] == 0,
        "use_fp16": settings["UseFp16"],
        "accelerator_device_index": settings["SelectedDeviceIndex"],
        "budget_limit": 0,
    }
)

print("settings", settings_parser.get_int("accelerator_device_index", 0), flush=True)

context = _ExecutorNodeContext(ProgressController(), settings_parser, Path())

gamma1icc = get_gamma_icc_profile()
dotgain20icc = get_dot20_icc_profile()

dotgain20togamma1transform = ImageCms.buildTransformFromOpenProfiles(
    dotgain20icc, gamma1icc, "L", "L"
)
gamma1todotgain20transform = ImageCms.buildTransformFromOpenProfiles(
    gamma1icc, dotgain20icc, "L", "L"
)

if __name__ == "__main__":
    spandrel_custom.install()
    # gc.disable() #TODO!!!!!!!!!!!!
    # Record the start time
    start_time = time.time()

    image_format = None
    if workflow["WebpSelected"]:
        image_format = "webp"
    elif workflow["PngSelected"]:
        image_format = "png"
    elif workflow["AvifSelected"]:
        image_format = "avif"
    else:
        image_format = "jpeg"

    target_scale: float | None = None
    target_width = 0
    target_height = 0

    grayscale_detection_threshold = workflow["GrayscaleDetectionThreshold"]

    if workflow["ModeScaleSelected"]:
        target_scale = workflow["UpscaleScaleFactor"]
    elif workflow["ModeWidthSelected"]:
        target_width = workflow["ResizeWidthAfterUpscale"]
    elif workflow["ModeHeightSelected"]:
        target_height = workflow["ResizeHeightAfterUpscale"]
    else:
        target_width = workflow["DisplayDeviceWidth"]
        target_height = workflow["DisplayDeviceHeight"]

    if workflow["SelectedTabIndex"] == 1:
        upscale_folder(
            workflow["InputFolderPath"],
            workflow["OutputFolderPath"],
            workflow["OutputFilename"],
            workflow["UpscaleImages"],
            workflow["UpscaleArchives"],
            workflow["OverwriteExistingFiles"],
            image_format,
            workflow["LossyCompressionQuality"],
            workflow["UseLosslessCompression"],
            target_scale,
            target_width,
            target_height,
            workflow["Chains"]["$values"],
            loaded_models,
            grayscale_detection_threshold,
        )
    elif workflow["SelectedTabIndex"] == 0:
        upscale_file(
            workflow["InputFilePath"],
            workflow["OutputFolderPath"],
            workflow["OutputFilename"],
            workflow["OverwriteExistingFiles"],
            image_format,
            workflow["LossyCompressionQuality"],
            workflow["UseLosslessCompression"],
            target_scale,
            target_width,
            target_height,
            workflow["Chains"]["$values"],
            loaded_models,
            grayscale_detection_threshold,
        )

    # # Record the end time
    end_time = time.time()

    # # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
