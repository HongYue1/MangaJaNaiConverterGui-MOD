"""Pixels in, bytes out: decoding, ICC transforms and encoding.

Everything here sits on the boundary between the pipeline's numpy arrays and
the outside world's files and byte buffers, and all of it shares the same set of
fallbacks, which is why it belongs in one module rather than next to the
transforms that consume the arrays:

- JPEG XL has three possible routes in and out (libvips, pillow-jxl, and the
  bundled cjxl/djxl executables). `encode_capabilities()` decides which one this
  install really has by encoding a 1x1 image, because a libvips build can
  advertise an operation it then fails to complete; `capabilities.vips_has()`
  alone is not enough.
- `no_window()` lives here because those JPEG XL tools are the only
  subprocesses the worker starts, and without the flag every page would flash a
  console window on Windows.

The heavy handles are read as `runtime.<name>` at call time and never imported
by value, so this module stays importable before the heavy stack is loaded.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

from janai.core.formats import FORMATS, merged, save_kwargs
from janai.worker import capabilities, runtime
from janai.worker.environment import PATHS
from janai.worker.events import log


# --------------------------------------------------------------------------- #
# encoder capability probe
# --------------------------------------------------------------------------- #
def encode_capabilities() -> dict:
    """What this install can really write, checked by encoding a 1x1 image."""
    caps: dict[str, dict] = {}
    probe = runtime.np.zeros((1, 1), dtype=runtime.np.uint8)
    for fid, spec in FORMATS.items():
        entry = {"ok": False, "via": "", "reason": ""}
        if capabilities.vips_has(spec.probe):
            try:
                vips_from_array(probe).write_to_buffer(spec.suffix)
                entry.update(ok=True, via="libvips")
            except Exception as exc:
                entry["reason"] = f"libvips {spec.probe}: {exc}"
        else:
            entry["reason"] = f"libvips has no {spec.probe}"
        if not entry["ok"] and fid == "jxl":
            if capabilities.pillow_jxl_available():
                entry.update(ok=True, via="pillow-jxl", reason="")
            elif capabilities.find_cjxl():
                entry.update(ok=True, via="cjxl", reason="")
        caps[fid] = entry
    return caps


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
def vips_from_array(arr):
    a = runtime.np.ascontiguousarray(arr)
    if a.ndim == 2:
        h, w = a.shape
        bands = 1
    else:
        h, w, bands = a.shape
    img = runtime.pyvips.Image.new_from_memory(a.tobytes(), w, h, bands, "uchar")
    try:
        img = img.copy(interpretation="b-w" if bands == 1 else "srgb")
    except Exception:
        pass
    return img


def read_image(path: Path):
    if path.suffix.lower() == ".jxl" and not capabilities.vips_has("jxlload"):
        return read_jxl_djxl(path)
    return (
        runtime.pyvips.Image.new_from_file(str(path), access="sequential", fail=True)
        .icc_transform("srgb")
        .numpy()
    )


def read_image_bytes(data: bytes, name: str = ""):
    if name.lower().endswith(".jxl") and not capabilities.vips_has("jxlload"):
        with tempfile.TemporaryDirectory(prefix="janai-jxl-") as td:
            src = Path(td) / "in.jxl"
            src.write_bytes(data)
            return read_jxl_djxl(src)
    return (
        runtime.pyvips.Image.new_from_buffer(data, "", access="sequential")
        .icc_transform("srgb")
        .numpy()
    )


def read_jxl_djxl(path: Path):
    """Decode JPEG XL through djxl, for a libvips built without jxlload."""
    exe = capabilities.find_djxl()
    if not exe:
        raise RuntimeError(
            "this libvips cannot read JPEG XL; put djxl.exe in the tools folder "
            "or on PATH to read .jxl inputs"
        )
    with tempfile.TemporaryDirectory(prefix="janai-jxl-") as td:
        dst = Path(td) / "decoded.png"
        run = subprocess.run(
            [exe, str(path), str(dst)], check=False, capture_output=True, creationflags=no_window()
        )
        if run.returncode != 0 or not dst.exists():
            raise RuntimeError(f"djxl failed: {run.stderr.decode('utf-8', 'replace')[:300]}")
        return (
            runtime.pyvips.Image.new_from_file(str(dst), access="sequential", fail=True)
            .icc_transform("srgb")
            .numpy()
        )


# Built once per process and then reused: opening the two profiles and building
# the transforms costs more than the resize that uses them. The empty tuple is a
# cached "these profiles are missing", so the warning is logged once rather than
# once per page.
_icc_pair = None
_icc_warned = False


def icc_transforms():
    """(dotgain20 -> gamma1, gamma1 -> dotgain20) or None when profiles are missing."""
    global _icc_pair, _icc_warned
    if _icc_pair is not None:
        return _icc_pair
    gamma = PATHS.icc("Custom Gray Gamma 1.0.icc")
    dot = PATHS.icc("Dot Gain 20%.icc")
    if not (gamma and dot):
        if not _icc_warned:
            log("grayscale ICC profiles missing, using Lanczos for the final resize", "warn")
            _icc_warned = True
        _icc_pair = ()
        return _icc_pair
    g = runtime.ImageCms.getOpenProfile(str(gamma))
    d = runtime.ImageCms.getOpenProfile(str(dot))
    _icc_pair = (
        runtime.ImageCms.buildTransformFromOpenProfiles(d, g, "L", "L"),
        runtime.ImageCms.buildTransformFromOpenProfiles(g, d, "L", "L"),
    )
    return _icc_pair


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #
def encode(image, fid: str, opts: dict, caps: dict) -> bytes:
    spec = FORMATS[fid]
    cap = caps.get(fid, {})
    via = cap.get("via") or "libvips"
    if via == "libvips":
        return encode_vips(image, fid, opts)
    if fid == "jxl" and via == "pillow-jxl":
        return encode_jxl_pillow(image, opts)
    if fid == "jxl" and via == "cjxl":
        return encode_jxl_cjxl(image, opts)
    raise RuntimeError(f"no encoder available for {spec.label}")


def encode_vips(image, fid: str, opts: dict) -> bytes:
    spec = FORMATS[fid]
    img = vips_from_array(image)
    if img.bands == 4 and fid == "jpeg":
        img = img.flatten(background=255)
    kwargs = save_kwargs(fid, opts)
    try:
        return img.write_to_buffer(spec.suffix, **kwargs)
    except Exception as exc:
        keep = {
            k: v for k, v in kwargs.items() if k in ("Q", "lossless", "compression", "distance")
        }
        log(f"{spec.label}: {exc}; retrying with {keep or 'defaults'}", "warn")
        return img.write_to_buffer(spec.suffix, **keep)


def _pil_image(image):
    if image.ndim == 2:
        return runtime.PILImage.fromarray(image, mode="L")
    if image.shape[2] == 4:
        return runtime.PILImage.fromarray(image, mode="RGBA")
    return runtime.PILImage.fromarray(image[:, :, :3], mode="RGB")


def encode_jxl_pillow(image, opts: dict) -> bytes:
    import pillow_jxl  # noqa: F401

    vals = merged("jxl", opts)
    kwargs: dict[str, Any] = {"effort": int(vals["effort"])}
    if vals.get("lossless"):
        kwargs["lossless"] = True
    else:
        if vals.get("rate_mode") == "distance":
            log("pillow-jxl has no distance control; using the quality value instead", "warn")
        kwargs["quality"] = int(vals["Q"])
    buf = BytesIO()
    _pil_image(image).save(buf, format="JXL", **kwargs)
    return buf.getvalue()


def encode_jxl_cjxl(image, opts: dict) -> bytes:
    exe = capabilities.find_cjxl()
    if not exe:
        raise RuntimeError("cjxl not found")
    vals = merged("jxl", opts)
    with tempfile.TemporaryDirectory(prefix="janai-jxl-") as td:
        src = Path(td) / "in.png"
        dst = Path(td) / "out.jxl"
        vips_from_array(image).write_to_file(str(src))
        cmd = [exe, str(src), str(dst), "-e", str(int(vals["effort"]))]
        if vals.get("lossless"):
            cmd += ["-d", "0", "--lossless_jpeg=0"]
        elif vals.get("rate_mode") == "distance":
            cmd += ["-d", str(float(vals["distance"]))]
        else:
            cmd += ["-q", str(int(vals["Q"]))]
        run = subprocess.run(cmd, check=False, capture_output=True, creationflags=no_window())
        if run.returncode != 0 or not dst.exists():
            raise RuntimeError(f"cjxl failed: {run.stderr.decode('utf-8', 'replace')[:300]}")
        return dst.read_bytes()


def no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
