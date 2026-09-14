"""Per-page pixel transforms: resize, grayscale detection, levels.

Everything here takes an array and returns an array. It is the work that
happens around the model, between decoding a page and encoding it again, and
it is grouped by that lifecycle rather than by which library each call reaches
for: the resize family, the gray/colour decision, and the tonal fixes that
only apply once a page has been called gray.

`predict_size()` lives next to `final_resize()` on purpose. It answers "what
size would `final_resize()` produce" without touching a pixel, which is what
the dry run reports to the GUI. Two copies of that arithmetic in two modules
would be free to disagree, and the user would see a dry run that promises a
size the real run does not deliver.

Heavy handles are read as `runtime.<name>` at call time, never imported by
value, so this module is importable before the imaging stack is loaded.
"""

from __future__ import annotations

from janai.worker import imageio, runtime


def standard_resize(image, new_size: tuple[int, int]):
    out = image.astype(runtime.np.float32) / 255.0
    out = runtime.cx_resize(out, new_size, runtime.ResizeFilter.Lanczos, False)
    out = (out * 255).round().astype(runtime.np.uint8)
    if runtime.get_h_w_c(image)[2] == 1 and out.ndim == 3:
        out = runtime.np.squeeze(out, axis=-1)
    return out


def dotgain20_resize(image, new_size: tuple[int, int]):
    pair = imageio.icc_transforms()
    if not pair:
        return standard_resize(image, new_size)
    to_gamma, to_dotgain = pair
    h = runtime.get_h_w_c(image)[0]
    size_ratio = h / max(1, new_size[1])
    blur = (1 / size_ratio - 1) / 3.5
    if blur >= 0.1:
        blur = min(blur, 250)
    pil = runtime.PILImage.fromarray(image, mode="L")
    pil = pil.filter(runtime.ImageFilter.GaussianBlur(radius=blur))
    pil = runtime.ImageCms.applyTransform(pil, to_gamma, False)
    out = runtime.np.array(pil).astype(runtime.np.float32) / 255.0
    out = runtime.cx_resize(out, new_size, runtime.ResizeFilter.CubicCatrom, False)
    out = (out * 255).round().astype(runtime.np.uint8)
    pil = runtime.PILImage.fromarray(out[:, :, 0] if out.ndim == 3 else out, mode="L")
    return runtime.np.array(runtime.ImageCms.applyTransform(pil, to_dotgain, False))


def image_resize(image, new_size: tuple[int, int], is_gray: bool):
    if is_gray and image.ndim == 2:
        return dotgain20_resize(image, new_size)
    return standard_resize(image, new_size)


GRAY_SAMPLE = 768  # long edge of the copy the grayscale test looks at


def gray_stats(image, threshold: float, colour_percent: float = 0.25) -> tuple[bool, float, float]:
    """(is_grayscale, mean colour excess, percent of clearly coloured pixels).

    Three things were wrong with the inherited test:

    * it averaged over every pixel of a full size page, which is slow on an
      8000 px scan and, worse, blind to a small but unmistakably coloured area:
      a title logo or one colour panel averages away to nothing, the page is
      called gray, and the colour is then squashed out of it for good;
    * it summed the three channel differences in uint8, so a genuinely colourful
      pixel could wrap past 255 back down to a small number and count as gray;
    * scanner and JPEG chroma noise pushed clean gray pages over the threshold,
      which is what made the setting feel arbitrary.

    So: measure on an area-averaged sample (fast, and averaging is what removes
    the chroma noise), sum in int32, and refuse to call a page gray when a
    non-trivial share of its pixels are properly coloured. The mean-excess
    metric and its ``threshold / 12`` comparison are kept, so an existing
    threshold still means the same thing.
    """
    h, w, c = runtime.hwc(image)
    if c == 1:
        return True, 0.0, 0.0

    sample = image[:, :, :3]
    long_edge = max(h, w)
    if long_edge > GRAY_SAMPLE:
        factor = GRAY_SAMPLE / float(long_edge)
        sample = runtime.cv2.resize(
            sample,
            (max(1, int(w * factor)), max(1, int(h * factor))),
            interpolation=runtime.cv2.INTER_AREA,
        )

    b, g, r = runtime.cv2.split(sample)
    t = int(max(0, min(255, round(threshold))))
    excess = (
        runtime.cv2.subtract(runtime.cv2.absdiff(r, g), t).astype(runtime.np.int32)
        + runtime.cv2.subtract(runtime.cv2.absdiff(r, b), t).astype(runtime.np.int32)
        + runtime.cv2.subtract(runtime.cv2.absdiff(g, b), t).astype(runtime.np.int32)
    )
    high = runtime.cv2.max(runtime.cv2.max(r, g), b)
    low = runtime.cv2.min(runtime.cv2.min(r, g), b)
    keep = ~runtime.np.logical_or(high == 0, low == 255)  # skip pure black / pure white
    kept = int(runtime.np.count_nonzero(keep))
    if kept == 0:
        return False, 0.0, 0.0

    mean_excess = float(excess[keep].sum()) / (kept * 3)
    spread = runtime.cv2.subtract(high, low)
    coloured = int(runtime.np.count_nonzero(runtime.np.logical_and(keep, spread > max(8, 2 * t))))
    percent = 100.0 * coloured / kept
    is_gray = mean_excess <= threshold / 12 and percent <= max(0.0, colour_percent)
    return is_gray, mean_excess, percent


def to_grayscale(image):
    c = runtime.get_h_w_c(image)[2]
    if c == 3:
        return runtime.cv2.cvtColor(image, runtime.cv2.COLOR_BGR2GRAY)
    if c == 4:
        return runtime.cv2.cvtColor(image, runtime.cv2.COLOR_BGRA2GRAY)
    return image


def auto_levels(image):
    """Black/white point stretch based on histogram peaks (grayscale only)."""
    pil = runtime.PILImage.fromarray(image).convert("L")
    hist = pil.histogram()

    black, peak = 0, hist[0]
    for i in range(1, 31):
        if hist[i] > peak:
            peak, black = hist[i], i
    run = 0
    for i in range(31, 256):
        if hist[i] > peak:
            run, peak, black = 0, hist[i], i
        elif hist[i] < peak:
            run += 1
            if run > 1:
                break

    white, peak = 255, hist[255]
    for i in range(254, 224, -1):
        if hist[i] > peak:
            peak, white = hist[i], i
    run = 0
    for i in range(223, -1, -1):
        if hist[i] > peak:
            run, peak, white = 0, hist[i], i
        elif hist[i] < peak:
            run += 1
            if run > 1:
                break

    if white <= black:
        return runtime.normalize(image)
    arr = runtime.np.array(pil).astype("float32")
    arr = runtime.np.maximum(arr - black, 0) / (white - black)
    return runtime.np.clip(arr, 0, 1)


def final_resize(image, scale: float, width: int, height: int, ow: int, oh: int, is_gray: bool):
    if height and width:
        if height / oh < width / ow:
            width = 0
        else:
            height = 0
    h, w = runtime.get_h_w_c(image)[:2]
    if height:
        if h != height:
            return image_resize(image, (round(w * height / h), height), is_gray)
    elif width:
        if w != width:
            return image_resize(image, (width, round(h * width / w)), is_gray)
    else:
        target = round(oh * scale)
        if h != target:
            return image_resize(image, (round(w * target / h), target), is_gray)
    return image


def predict_size(ow: int, oh: int, t_scale: float, t_w: int, t_h: int) -> tuple[int, int]:
    """The size final_resize() lands on, without touching a pixel."""
    if t_w and t_h:  # fit: whichever side runs out first wins
        if t_h / max(1, oh) < t_w / max(1, ow):
            t_w = 0
        else:
            t_h = 0
    if t_h:
        return max(1, round(ow * t_h / max(1, oh))), t_h
    if t_w:
        return t_w, max(1, round(oh * t_w / max(1, ow)))
    return max(1, round(ow * t_scale)), max(1, round(oh * t_scale))


def is_long_strip(w: int, h: int, max_side: int, min_aspect: float, min_pixels: int) -> bool:
    """True for webtoon-style mega strips that may be passed through untouched.

    Adopted from the other fork's SkipLargeLong* settings, with one deliberate
    change: the pixel clause also requires the long aspect, so an ordinary
    large page (a 4000x2400 spread, say) is never mistaken for a strip.
    """
    if w <= 0 or h <= 0:
        return False
    aspect = max(w, h) / max(1, min(w, h))
    if aspect < min_aspect:
        return False
    return max(w, h) >= max_side or (w * h) >= min_pixels
