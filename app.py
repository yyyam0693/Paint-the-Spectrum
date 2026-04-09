"""
Paint the Spectrum: An Interactive 2D Fourier Learning Tool
-----------------------------------------------------------

Lets a user:
  1. upload an image (the spatial-domain view),
  2. see the same image as a 2D Fourier Transform (the frequency-domain view),
  3. edit the spectrum by painting or by applying canonical filters, and
  4. watch the inverse 2D Fourier Transform rebuild the image.

Design notes
------------
- Mask edits are applied to the *complex* shifted FFT, not to the displayed
  log-magnitude picture. The log-magnitude is only used for visualisation.
- We reshape masks symmetrically where needed (notch pairs) so the inverse
  transform remains essentially real, thanks to conjugate symmetry of the
  Fourier transform of a real-valued image.
- Working images are downsized (longest side <= max_dim) for interactive speed.
"""

from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# =============================================================================
# Page configuration
# =============================================================================
st.set_page_config(page_title="Paint the Spectrum", layout="wide")


# =============================================================================
# Image loading & preprocessing
# =============================================================================
def load_image(uploaded_file) -> Image.Image | None:
    """Load an uploaded file into a PIL RGB image. Returns None on failure."""
    if uploaded_file is None:
        return None
    try:
        return Image.open(uploaded_file).convert("RGB")
    except Exception:
        return None


def get_sample_image(size: int = 256) -> Image.Image:
    """
    Deterministic sample image with rich frequency content: a smooth gradient
    (low-frequency dominant), diagonal stripes (a clear directional frequency),
    a filled circle (edges -> broadband), and a few bright dots (point-like
    features that spread energy across the spectrum).
    """
    x = np.linspace(-1.0, 1.0, size)
    y = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(x, y)

    base = 0.5 + 0.3 * np.cos(2 * np.pi * xx * 0.5) * np.cos(2 * np.pi * yy * 0.5)
    stripes = 0.15 * np.sign(np.sin(2 * np.pi * 8 * (xx + yy)))
    r = np.sqrt(xx ** 2 + yy ** 2)
    circle = np.where(r < 0.32, 0.35, 0.0)
    dots = np.zeros_like(base)
    for cx, cy in [(0.6, 0.55), (-0.55, 0.35), (0.4, -0.7)]:
        dots += np.where(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) < 0.03, 0.5, 0.0)

    img = np.clip(base + stripes + circle + dots, 0.0, 1.0)
    arr = (img * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def preprocess_image(pil_image: Image.Image, max_dim: int = 256) -> np.ndarray:
    """
    Convert to grayscale, resize (aspect-preserving) so the longer side equals
    ``max_dim``, and return a float32 array in [0, 1].
    """
    gray = pil_image.convert("L")
    w, h = gray.size
    longer = max(w, h)
    if longer > max_dim:
        scale = max_dim / float(longer)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        gray = gray.resize((new_w, new_h), Image.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    return arr


# =============================================================================
# Fourier primitives
# =============================================================================
def compute_fft2(image_array: np.ndarray):
    """Return (F, shifted_F) as complex arrays."""
    F = np.fft.fft2(image_array)
    Fs = np.fft.fftshift(F)
    return F, Fs


def make_log_spectrum(shifted_fft: np.ndarray) -> np.ndarray:
    """log(1 + |G|) rescaled into [0, 1] for visualisation."""
    mag = np.abs(shifted_fft)
    log_mag = np.log1p(mag)
    mn, mx = float(log_mag.min()), float(log_mag.max())
    if mx - mn < 1e-12:
        return np.zeros_like(log_mag, dtype=np.float32)
    return ((log_mag - mn) / (mx - mn)).astype(np.float32)


def apply_mask_and_reconstruct(shifted_fft: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Multiply the shifted complex spectrum by the mask, invert the shift,
    run inverse FFT, and return the real part.  The result is real up to
    numerical noise when the mask is conjugate-symmetric (as all presets are).
    """
    modified_shifted = shifted_fft * mask
    modified = np.fft.ifftshift(modified_shifted)
    recon = np.fft.ifft2(modified)
    return np.real(recon)


def normalize_for_display(arr: np.ndarray) -> np.ndarray:
    """Rescale to [0, 1] for matplotlib imshow."""
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


# =============================================================================
# Mask builders (all in shifted-FFT coordinates — center = (H//2, W//2))
# =============================================================================
def _centered_grid(shape):
    h, w = shape
    cy, cx = h // 2, w // 2
    y = np.arange(h).reshape(-1, 1) - cy
    x = np.arange(w).reshape(1, -1) - cx
    return y, x


def make_low_pass_mask(shape, radius: float) -> np.ndarray:
    y, x = _centered_grid(shape)
    r = np.sqrt(y ** 2 + x ** 2)
    return (r <= radius).astype(np.float32)


def make_high_pass_mask(shape, radius: float) -> np.ndarray:
    y, x = _centered_grid(shape)
    r = np.sqrt(y ** 2 + x ** 2)
    return (r > radius).astype(np.float32)


def make_band_pass_mask(shape, inner_radius: float, outer_radius: float) -> np.ndarray:
    y, x = _centered_grid(shape)
    r = np.sqrt(y ** 2 + x ** 2)
    inner = min(inner_radius, outer_radius)
    outer = max(inner_radius, outer_radius)
    return ((r >= inner) & (r <= outer)).astype(np.float32)


def make_notch_mask(shape, offset_x: int, offset_y: int, radius: float) -> np.ndarray:
    """
    Remove two small disks: at (cy+oy, cx+ox) and its conjugate-symmetric
    partner (cy-oy, cx-ox).  The pair keeps the reconstruction real.
    """
    h, w = shape
    cy, cx = h // 2, w // 2
    y = np.arange(h).reshape(-1, 1)
    x = np.arange(w).reshape(1, -1)
    r1 = np.sqrt((y - (cy + offset_y)) ** 2 + (x - (cx + offset_x)) ** 2)
    r2 = np.sqrt((y - (cy - offset_y)) ** 2 + (x - (cx - offset_x)) ** 2)
    mask = np.ones(shape, dtype=np.float32)
    mask[r1 <= radius] = 0.0
    mask[r2 <= radius] = 0.0
    return mask


def make_directional_strip_mask(shape, orientation: str, width: int) -> np.ndarray:
    """
    Remove a horizontal or vertical band that runs through DC.
    A horizontal strip mainly kills vertical-frequency components and
    therefore weakens horizontal stripes/textures in the spatial image.
    """
    h, w = shape
    cy, cx = h // 2, w // 2
    mask = np.ones(shape, dtype=np.float32)
    half = width / 2.0
    if orientation == "horizontal":
        y = np.arange(h).reshape(-1, 1)
        band = np.abs(y - cy) <= half
        mask = np.where(np.broadcast_to(band, shape), 0.0, 1.0).astype(np.float32)
    else:  # vertical
        x = np.arange(w).reshape(1, -1)
        band = np.abs(x - cx) <= half
        mask = np.where(np.broadcast_to(band, shape), 0.0, 1.0).astype(np.float32)
    return mask


# =============================================================================
# Educational text helpers
# =============================================================================
def explain_mask_effect(mask: np.ndarray, shifted_fft: np.ndarray) -> str:
    """
    Explain the mask effect using energy-weighted frequency regions instead of
    a plain average radius. Handles multiple removed regions better.

    Regions:
      - low:  r_norm < 0.20
      - mid:  0.20 <= r_norm <= 0.50
      - high: r_norm > 0.50

    Weighting:
      - uses |G[k,l]| as a proxy for importance
      - so removing a bright low-frequency patch counts more than removing a
        large but dim outer patch
    """
    h, w = mask.shape
    cy, cx = h // 2, w // 2
    y, x = _centered_grid((h, w))
    r = np.sqrt(y ** 2 + x ** 2)
    r_max = float(np.sqrt(cy ** 2 + cx ** 2))
    r_norm = r / max(r_max, 1e-9)

    removed = mask < 0.5
    if not removed.any():
        return "No frequencies were removed — the reconstruction matches the original."

    # Pixel coverage (same idea as before, but not used alone)
    coverage = float(removed.mean()) * 100.0

    # Energy weighting from the original shifted FFT
    mag = np.abs(shifted_fft).astype(np.float64)
    total_energy = mag.sum()

    removed_energy = mag[removed].sum()
    removed_energy_pct = 100.0 * removed_energy / max(total_energy, 1e-12)

    low = r_norm < 0.20
    mid = (r_norm >= 0.20) & (r_norm <= 0.50)
    high = r_norm > 0.50

    low_e = mag[removed & low].sum()
    mid_e = mag[removed & mid].sum()
    high_e = mag[removed & high].sum()

    bucket_total = low_e + mid_e + high_e + 1e-12
    low_share = low_e / bucket_total
    mid_share = mid_e / bucket_total
    high_share = high_e / bucket_total

    # Primary label = dominant removed region by energy
    shares = {
        "low": low_share,
        "mid": mid_share,
        "high": high_share,
    }
    dominant = max(shares, key=shares.get)

    # Build explanation text
    if dominant == "low":
        main = (
            "The removal is dominated by **low frequencies near the center**, "
            "so broad smooth structure and overall intensity patterns are most affected."
        )
    elif dominant == "high":
        main = (
            "The removal is dominated by **high frequencies farther from the center**, "
            "so fine detail and sharp edges are most affected."
        )
    else:
        main = (
            "The removal is dominated by **mid-range frequencies**, "
            "so medium-scale texture and repeating patterns are most affected."
        )

    # Mention meaningful secondary regions too
    secondary = []
    if dominant != "low" and low_share > 0.20:
        secondary.append("it includes some low-frequency removal")
    if dominant != "mid" and mid_share > 0.20:
        secondary.append("it includes some mid-range removal")
    if dominant != "high" and high_share > 0.20:
        secondary.append("it includes some high-frequency removal")

    if secondary:
        main += " In addition, " + ", and ".join(secondary) + "."

    return (
        f"Removed {coverage:.1f}% of the spectrum area, corresponding to "
        f"{removed_energy_pct:.1f}% of the spectrum magnitude. {main}"
    )


def explain_preset(name: str) -> str:
    texts = {
        "low_pass": (
            "**Low-pass filter** (Chapter 7). Keeps the low-frequency "
            "components near the centre and zeros out everything outside a radius. "
            "Low frequencies describe slow, smooth variation in brightness, so the "
            "output is a **smoother / blurrier** image — a frequency-domain blur. "
            "This is the classical Gaussian / average-filter story, but expressed "
            "directly on the spectrum."
        ),
        "high_pass": (
            "**High-pass filter**. The mirror image of a "
            "low-pass filter: it **removes the DC term and the low frequencies** "
            "and keeps only the outer ring. Since the low frequencies carry the "
            "overall brightness and slow variation, what remains is essentially "
            "the **edges and sharp transitions** — this is what an edge detector "
            "(Sobel, Laplacian) is doing, viewed from the frequency side."
        ),
        "band_pass": (
            "**Band-pass filter**. Keeps only a ring of "
            "frequencies between an inner and outer radius. Very low frequencies "
            "(the overall brightness) **and** very high frequencies (fine detail) "
            "are both discarded — only a specific *band* of spatial scales "
            "survives, so periodic textures of a particular size are emphasised."
        ),
        "notch": (
            "**Notch filter.** Removes a small, targeted pair of frequency "
            "components — the classic tool for killing periodic interference "
            "(scan lines, moiré, 50/60 Hz stripes). **Why a pair?** Because our "
            "image is real-valued, its Fourier transform obeys **conjugate "
            "symmetry**: $G[-k,\\,-l] = G^*[k,\\,l]$ (Chapter 4, DFT "
            "properties). If we zeroed only one bin, its conjugate partner on "
            "the opposite side of DC would survive and the inverse FFT would "
            "produce a complex image. Removing the **symmetric pair** keeps the "
            "reconstruction real."
        ),
        "directional": (
            "**Directional strip.** A band of frequencies aligned with one "
            "orientation is suppressed — a horizontal strip through DC kills "
            "the *vertical* frequencies, which removes *horizontal* stripe "
            "patterns in the image (and vice versa). Handy for cleaning up "
            "scan-line artifacts and other strongly-oriented textures."
        ),
    }
    return texts.get(name, "")


# =============================================================================
# Streamlit-specific render helpers
# =============================================================================
def show_image(img, caption: str | None = None, cmap: str | None = "gray"):
    fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=110)
    if cmap is None:
        ax.imshow(img)
    else:
        ax.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0)
    if caption:
        ax.set_title(caption, fontsize=10)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    st.pyplot(fig)
    plt.close(fig)


def show_spectrum(spec, caption: str | None = None, overlay_mask: np.ndarray | None = None):
    fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=110)
    ax.imshow(spec, cmap="magma", vmin=0.0, vmax=1.0)
    if overlay_mask is not None:
        removed = (overlay_mask < 0.5).astype(np.float32)
        zero = np.zeros_like(removed)
        ax.imshow(np.dstack([removed, zero, zero, 0.40 * removed]))
    if caption:
        ax.set_title(caption, fontsize=10)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    st.pyplot(fig)
    plt.close(fig)


def spectrum_to_pil(spec: np.ndarray) -> Image.Image:
    """Render the log-magnitude spectrum through the magma colormap as an RGB PIL image."""
    cmap = plt.get_cmap("magma")
    rgba = cmap(np.clip(spec, 0.0, 1.0))
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def pixels_from_plotly_selection(selection, shape) -> np.ndarray:
    """
    Convert a Plotly ``selection`` dict (as returned by ``st.plotly_chart`` with
    ``on_select='rerun'``) into a binary (H, W) mask of newly-painted pixels.

    * box selection → rectangular fill
    * lasso         → polygon fill
    """
    H, W = shape
    m = np.zeros((H, W), dtype=np.float32)
    if not isinstance(selection, dict):
        return m

    # Rectangle fills for box selections.
    for b in selection.get("box") or []:
        xs, ys = b.get("x", []), b.get("y", [])
        if len(xs) >= 2 and len(ys) >= 2:
            x0, x1 = sorted([int(round(xs[0])), int(round(xs[1]))])
            y0, y1 = sorted([int(round(ys[0])), int(round(ys[1]))])
            x0 = max(0, x0); x1 = min(W - 1, x1)
            y0 = max(0, y0); y1 = min(H - 1, y1)
            if x1 >= x0 and y1 >= y0:
                m[y0:y1 + 1, x0:x1 + 1] = 1.0

    # Polygon fills for lasso selections.
    lassos = selection.get("lasso") or []
    if lassos:
        from matplotlib.path import Path as MplPath
        yy, xx = np.mgrid[:H, :W]
        grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
        for lz in lassos:
            xs, ys = lz.get("x", []), lz.get("y", [])
            if len(xs) >= 3:
                poly = MplPath(np.column_stack([xs, ys]))
                inside = poly.contains_points(grid_pts).reshape(H, W)
                m[inside] = 1.0
    return m


# =============================================================================
# Session-state management
# =============================================================================
def _image_signature(pil_image: Image.Image | None, max_dim: int):
    if pil_image is None:
        return None
    return (pil_image.size, pil_image.mode, int(max_dim), hash(pil_image.tobytes()))


def set_active_image(pil_image: Image.Image, max_dim: int) -> None:
    """Store a new active image and all derived Fourier quantities in session state."""
    working = preprocess_image(pil_image, max_dim=max_dim)
    F, Fs = compute_fft2(working)
    log_spec = make_log_spectrum(Fs)

    st.session_state.original_image = pil_image
    st.session_state.working_image = working
    st.session_state.fft = F
    st.session_state.shifted_fft = Fs
    st.session_state.log_spectrum = log_spec
    st.session_state.mask = np.ones_like(working, dtype=np.float32)
    st.session_state.painted_mask = np.zeros_like(working, dtype=np.float32)
    st.session_state.last_selection_sig = None
    st.session_state.reconstructed = working.copy()
    st.session_state.image_signature = _image_signature(pil_image, max_dim)
    # bump the plotly widget key so the selection clears when the image changes
    st.session_state.canvas_key = st.session_state.get("canvas_key", 0) + 1


def ensure_session_state():
    defaults = {
        "original_image": None,
        "working_image": None,
        "fft": None,
        "shifted_fft": None,
        "log_spectrum": None,
        "mask": None,
        "painted_mask": None,
        "last_selection_sig": None,
        "reconstructed": None,
        "image_signature": None,
        "canvas_key": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


ensure_session_state()


# =============================================================================
# Top-of-app layout
# =============================================================================
st.title("Paint the Spectrum")
st.subheader("An interactive 2D Fourier learning tool")
st.markdown(
    "An image lives in the **spatial domain** (pixels), but the *same* image can be "
    "described in the **frequency domain** using the 2D Fourier Transform. In the "
    "centred spectrum, the **centre** is the low-frequency region (slow, smooth "
    "brightness changes) and points **farther from the centre** are higher "
    "frequencies (edges, fine detail, sharp transitions). Upload an image once below, "
    "then explore all three tabs — edit the spectrum, and the inverse FFT rebuilds "
    "the image in front of you."
)

col_up, col_res = st.columns([3, 1])
with col_up:
    uploaded = st.file_uploader(
        "Upload an image (PNG / JPG / JPEG). If nothing is uploaded, a built-in sample is used.",
        type=["png", "jpg", "jpeg"],
    )
with col_res:
    max_dim = st.select_slider(
        "Working resolution (longest side, px)",
        options=[128, 192, 256, 384, 512, 768, 1024],
        value=256,
        help=(
            "Image is downsized so its longest side matches this. "
            "The FFT itself stays fast at every setting — the bottleneck is "
            "Plotly's interactive heatmap + box/lasso selection in Tab 2, "
            "which gets noticeably laggier past ~512 px. "
            "Use 256 for snappy painting; bump to 768–1024 if you want to "
            "see a crisp reconstruction on a real photo."
        ),
    )

sample_clicked = st.button("Use sample image")

# Decide which image this run should treat as active.
new_pil: Image.Image | None = None
if uploaded is not None:
    loaded = load_image(uploaded)
    if loaded is None:
        st.error("Could not read that file. Please upload a valid PNG or JPG image.")
    else:
        if max(loaded.size) > 1024:
            st.info(
                f"Uploaded image is {loaded.size[0]} x {loaded.size[1]}; "
                f"it will be downsized to {max_dim}px on the longest side for Fourier processing."
            )
        new_pil = loaded
elif sample_clicked or st.session_state.original_image is None:
    new_pil = get_sample_image(size=256)
else:
    # Keep whatever the user last loaded; the resolution slider may still have changed.
    new_pil = st.session_state.original_image

# (Re)compute whenever image *or* working resolution changes. The signature
# includes max_dim, so resolution changes alone also trigger a refresh.
target_sig = _image_signature(new_pil, max_dim)
if target_sig is not None and target_sig != st.session_state.image_signature:
    set_active_image(new_pil, max_dim)

st.caption(
    "The same loaded image is used across all three tabs. "
    "Change the uploader or the working-resolution control above to reload it."
)

# Convenience handles (read-only views into session state).
working = st.session_state.working_image
shifted_fft = st.session_state.shifted_fft
log_spectrum = st.session_state.log_spectrum
H, W = working.shape


# =============================================================================
# Tabs
# =============================================================================
tab1, tab2, tab3 = st.tabs(
    ["1. Original & Spectrum", "2. Paint the Spectrum", "3. Frequency Surgery Presets"]
)


# --------------------------- TAB 1: Original & Spectrum ---------------------
with tab1:
    st.markdown(
        "#### Moving between spatial and frequency domains\n"
        "Below you see the image you uploaded, its grayscale working copy, and its "
        "centred log-magnitude spectrum. The **centre** of the spectrum holds the "
        "**low frequencies** (slow, smooth brightness changes); points **farther "
        "from the centre** are **higher frequencies** (edges, fine detail, sharp "
        "transitions). A 2D Fourier Transform is simply the FFT applied to a 2D image."
    )

    show_dims = st.checkbox("Show image dimensions", value=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        rgb_arr = np.asarray(st.session_state.original_image.convert("RGB")) / 255.0
        show_image(rgb_arr, caption="Original (uploaded)", cmap=None)
        if show_dims:
            ow, oh = st.session_state.original_image.size
            st.caption(f"Original size: {ow} × {oh}")
    with c2:
        show_image(working, caption="Grayscale working image")
        if show_dims:
            st.caption(f"Working size: {W} × {H}")
    with c3:
        show_spectrum(log_spectrum, caption="Log-magnitude spectrum (shifted)")
        if show_dims:
            st.caption(f"Spectrum size: {W} × {H}")

    # ---- DC component readout (Chapter 4: 'DC' = direct-current = zero frequency) ----
    st.markdown("#### The DC component: what is that bright dot at the centre?")
    st.markdown(
        "In the centred spectrum, the **brightest pixel at the centre** is the "
        "**DC component** — short for *direct current*, borrowed from electrical "
        "engineering (Chapter 4). It is the **zero-frequency** coefficient, and for "
        "a real image it is literally the **sum of every pixel value**. "
        "Equivalently, it equals $H \\cdot W \\cdot \\text{mean brightness}$. "
        "The three numbers below should all be (numerically) equal — that identity "
        "is why the centre pixel is so much brighter than anything else, and why we "
        "have to display $\\log(1 + |G|)$ instead of raw magnitude."
    )
    dc_value = float(np.abs(shifted_fft[H // 2, W // 2]))
    pixel_sum = float(working.sum())
    mean_times_N = float(working.mean()) * H * W
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("|G[cy, cx]|  (DC magnitude)", f"{dc_value:,.3f}")
    with m2:
        st.metric("Σ pixel values", f"{pixel_sum:,.3f}")
    with m3:
        st.metric("H · W · mean brightness", f"{mean_times_N:,.3f}")
    st.caption(
        f"Centre index = (cy, cx) = ({H // 2}, {W // 2}).  "
        f"Mean pixel value = {float(working.mean()):.4f} (on the 0–1 scale)."
    )

    # ---- Interactive frequency-bin readout (Chapter 5: cycles per image width/height) ----
    st.markdown("#### Hover the spectrum to read frequency bins")
    st.markdown(
        "Hover over any pixel of the spectrum below. The display shows the pixel "
        "index **(x, y)** *and* the frequency it represents: **$k$ cycles per "
        "image width** horizontally and **$l$ cycles per image height** "
        "vertically (Chapter 5: the spectrum is indexed by $G[k,\\,l]$, with "
        "$k$ running horizontally and $l$ running vertically). The bin at the "
        "centre is $(0, 0)$ = DC, and each step of 1 bin outward adds one full "
        "cycle across the image. A point at $(k, l) = (3, 0)$, for example, "
        "corresponds to a horizontal sinusoid that completes **3 full cycles "
        "across the image width**."
    )
    u_bins = (np.arange(W) - W // 2).reshape(1, -1)
    v_bins = (np.arange(H) - H // 2).reshape(-1, 1)
    UU = np.broadcast_to(u_bins, (H, W))
    VV = np.broadcast_to(v_bins, (H, W))
    bin_custom = np.stack([UU, VV], axis=-1)

    fig_hover = go.Figure()
    fig_hover.add_trace(go.Heatmap(
        z=log_spectrum,
        customdata=bin_custom,
        colorscale="Magma",
        showscale=False,
        hovertemplate=(
            "pixel (x, y) = (%{x}, %{y})<br>"
            "k = %{customdata[0]} cycles / image width<br>"
            "l = %{customdata[1]} cycles / image height<br>"
            "log(1 + |G|) = %{z:.3f}"
            "<extra></extra>"
        ),
    ))
    fig_hover.update_layout(
        width=440, height=440,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, range=[-0.5, W - 0.5], constrain="domain"),
        yaxis=dict(visible=False, range=[H - 0.5, -0.5],
                   scaleanchor="x", scaleratio=1.0),
        plot_bgcolor="#111",
        paper_bgcolor="#111",
    )
    st.plotly_chart(fig_hover, use_container_width=False, key="tab1_hover_spectrum")
    st.caption(
        "**Symmetry check (Chapter 4).** Because the image is real-valued, its "
        "Fourier transform obeys $G[-k,\\,-l] = G^*[k,\\,l]$, so the "
        "magnitude spectrum is point-symmetric about the centre. Hover a bright "
        "spot and its opposite across DC — their magnitudes should match."
    )

    with st.expander("Advanced view"):
        show_raw = st.checkbox("Show raw magnitude vs log magnitude", value=False)
        show_unshifted = st.checkbox("Show the unshifted spectrum (DC at a corner)", value=False)

        if show_raw:
            raw_norm = normalize_for_display(np.abs(shifted_fft))
            cc1, cc2 = st.columns(2)
            with cc1:
                show_spectrum(raw_norm, caption="Raw magnitude")
                st.caption(
                    "Raw magnitude is dominated by the extremely bright DC term, so "
                    "visually it hides everything else. That is why we display log "
                    "magnitude instead."
                )
            with cc2:
                show_spectrum(log_spectrum, caption="Log magnitude")
                st.caption("log(1 + |G|) compresses the huge dynamic range so structure is visible.")

        if show_unshifted:
            unshifted_spec = make_log_spectrum(st.session_state.fft)
            show_spectrum(unshifted_spec, caption="Unshifted spectrum")
            st.caption(
                "Without fftshift, the zero-frequency (DC) component sits in the corner. "
                "fftshift moves DC to the centre, which is much easier to interpret visually."
            )


# --------------------------- TAB 2: Paint the Spectrum ----------------------
with tab2:
    st.markdown(
        "#### Paint on the spectrum to modify frequencies\n"
        "Use the Plotly toolbar on the top-right of the spectrum to pick a tool "
        "(**Box Select**, **Lasso Select**, or just click a point) and mark regions "
        "of the spectrum. Each selection is **added** to the painted area. "
        "The painted area is **removed** from the frequency representation by default, "
        "and the image is rebuilt with the inverse 2D Fourier Transform. Try painting "
        "near the centre vs. near the edges to see the difference."
    )

    cc_controls, cc_plot = st.columns([1, 2])
    with cc_controls:
        mode = st.radio(
            "Brush effect",
            ["Remove painted frequencies", "Keep only painted frequencies"],
            index=0,
        )
        if st.button("Reset painted area", use_container_width=True):
            st.session_state.painted_mask = np.zeros_like(working, dtype=np.float32)
            st.session_state.last_selection_sig = None
            st.session_state.canvas_key += 1
            st.rerun()

        st.markdown("---")
        st.caption(
            "**Tip:** painted regions define a **binary mask** over the shifted "
            "Fourier domain. That mask multiplies the *complex* spectrum before "
            "the inverse transform, so this is a genuine frequency-domain edit, "
            "not a photo filter."
        )

    plot_key = f"spectrum_plotly_{st.session_state.canvas_key}"

    # IMPORTANT: fold any pending selection from the *previous* render into
    # ``painted_mask`` BEFORE we build the figure, so the red overlay we draw
    # reflects the click that just happened (not the one before it).
    prev_event = st.session_state.get(plot_key)
    selection = None
    if prev_event is not None:
        if hasattr(prev_event, "selection"):
            selection = prev_event.selection
        elif isinstance(prev_event, dict) and "selection" in prev_event:
            selection = prev_event["selection"]

    if selection is not None:
        sig = repr({
            "box": selection.get("box") or [],
            "lasso": selection.get("lasso") or [],
        })
        if sig != st.session_state.last_selection_sig:
            new_paint = pixels_from_plotly_selection(selection, (H, W))
            if new_paint.any():
                st.session_state.painted_mask = np.maximum(
                    st.session_state.painted_mask, new_paint
                )
            st.session_state.last_selection_sig = sig

    with cc_plot:
        display_size = 520
        fig = go.Figure()
        # Background: log-magnitude spectrum (magma colormap).
        fig.add_trace(go.Heatmap(
            z=log_spectrum,
            colorscale="Magma",
            showscale=False,
            hovertemplate="x=%{x}<br>y=%{y}<br>log|G|=%{z:.3f}<extra></extra>",
        ))
        # Overlay: the accumulated painted mask, in translucent red.
        fig.add_trace(go.Heatmap(
            z=st.session_state.painted_mask,
            colorscale=[[0.0, "rgba(0,0,0,0)"], [1.0, "rgba(255,60,60,0.55)"]],
            showscale=False,
            hoverinfo="skip",
            zmin=0.0, zmax=1.0,
        ))
        fig.update_layout(
            width=display_size, height=display_size,
            margin=dict(l=0, r=0, t=0, b=0),
            dragmode="select",
            xaxis=dict(visible=False, range=[-0.5, W - 0.5], constrain="domain"),
            yaxis=dict(visible=False, range=[H - 0.5, -0.5],
                       scaleanchor="x", scaleratio=1.0),
            plot_bgcolor="#111",
            paper_bgcolor="#111",
        )

        st.plotly_chart(
            fig,
            key=plot_key,
            on_select="rerun",
            selection_mode=("points", "box", "lasso"),
            use_container_width=False,
        )
        st.caption(
            "Use the **Box Select** / **Lasso Select** tools in the Plotly toolbar, "
            "or click single points. Each selection accumulates in the red overlay."
        )

    painted = st.session_state.painted_mask
    if mode.startswith("Remove"):
        mask = (1.0 - painted).astype(np.float32)
    else:
        mask = painted.astype(np.float32)
    st.session_state.mask = mask

    recon = apply_mask_and_reconstruct(shifted_fft, mask)
    recon_disp = normalize_for_display(recon)

    st.markdown("---")
    cA, cB, cC = st.columns(3)
    with cA:
        show_spectrum(log_spectrum * mask, caption="Masked spectrum")
        st.caption("Black areas were removed from the frequency representation.")
    with cB:
        show_spectrum(log_spectrum, caption="Spectrum + mask overlay", overlay_mask=mask)
        st.caption("Red overlay shows the regions that were removed.")
    with cC:
        show_image(recon_disp, caption="Reconstructed image")
        st.caption("Inverse FFT of the edited spectrum.")

    st.info(explain_mask_effect(mask, shifted_fft))

    # ---- The convolution theorem (Chapter 6) ----
    st.markdown("---")
    st.markdown("#### The same edit, expressed as a spatial-domain filter")
    st.markdown(
        "The **convolution theorem** (Chapter 6) says that **multiplication in the "
        "frequency domain is equivalent to convolution in the spatial domain**:\n\n"
        "$$\\;g[u,v] \\;*\\; h[u,v] \\;\\;\\longleftrightarrow\\;\\; G[k,l] \\cdot H[k,l]$$\n\n"
        "So when you painted a mask and multiplied it into the spectrum, that is "
        "**exactly the same** as convolving the original image with some spatial "
        "kernel $h[u,v]$. The kernel is simply the inverse 2D Fourier transform of "
        "your mask. Below you can see what that kernel looks like — this is the "
        "\"hidden\" filter your frequency-domain edit is secretly applying in "
        "pixel space."
    )

    # kernel lives in shifted (centred) coords like mask does
    kernel = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mask))))
    kernel_disp = normalize_for_display(kernel)
    k_abs_max = float(np.max(np.abs(kernel)))
    nonzero = float((mask > 0.5).sum())

    cK1, cK2 = st.columns([1, 1])
    with cK1:
        show_image(kernel_disp, caption="Equivalent spatial kernel  h[u, v]")
        st.caption(
            f"Peak |h| = {k_abs_max:.4g}  ·  kept-frequency pixels = {int(nonzero)} / {H * W}.  "
            "A mask that keeps only the centre gives a broad, smooth kernel "
            "(blurring = local averaging). A mask that keeps only the outer ring "
            "gives an oscillating kernel (edge detection = local differencing)."
        )
    with cK2:
        st.markdown(
            "**How to read this picture.** Each pixel of the kernel tells you "
            "how much the corresponding *neighbour* of a pixel contributes to the "
            "output. Bright centre + dark surround = high-pass (edge detector). "
            "A smooth bright blob = low-pass (blur). Oscillating stripes = "
            "band-pass (picks out a specific scale / orientation).\n\n"
            "**Takeaway.** Every spatial filter you have ever seen (Gaussian "
            "blur, Sobel edge filter, sharpening, unsharp mask…) corresponds to "
            "*some* shape of mask in the frequency domain, and vice versa — "
            "that is the content of the convolution theorem."
        )


# --------------------------- TAB 3: Frequency Surgery Presets ---------------
with tab3:
    st.markdown(
        "#### Frequency surgery presets\n"
        "These are the canonical frequency-domain filters  — "
        "low-pass from **Chapter 7**, high-pass, band-pass — plus the **notch** and **directional "
        "strip** used to clean up periodic artifacts. Choose a preset, tune its "
        "parameters, and watch the mask and the reconstruction update. Compared "
        "with free-form painting in Tab 2, these are the standard filters you "
        "would actually name in a report or a paper."
    )

    preset = st.selectbox(
        "Preset",
        ["Low-pass", "High-pass", "Band-pass", "Notch filter", "Directional strip"],
    )

    r_max = int(np.ceil(np.sqrt((H / 2) ** 2 + (W / 2) ** 2)))
    mask_preset = np.ones((H, W), dtype=np.float32)
    explanation_key = ""

    if preset == "Low-pass":
        radius = st.slider("Low-pass radius (px)", 1, r_max, max(1, r_max // 6))
        mask_preset = make_low_pass_mask((H, W), radius)
        explanation_key = "low_pass"
    elif preset == "High-pass":
        radius = st.slider("High-pass radius (px)", 1, r_max, max(1, r_max // 6))
        mask_preset = make_high_pass_mask((H, W), radius)
        explanation_key = "high_pass"
    elif preset == "Band-pass":
        cA, cB = st.columns(2)
        with cA:
            inner = st.slider("Inner radius (px)", 0, r_max, max(1, r_max // 8))
        with cB:
            outer = st.slider("Outer radius (px)", 1, r_max, max(2, r_max // 3))
        mask_preset = make_band_pass_mask((H, W), inner, outer)
        explanation_key = "band_pass"
    elif preset == "Notch filter":
        cA, cB, cC = st.columns(3)
        with cA:
            offset_x = st.slider("Notch offset X", -W // 2, W // 2, max(1, W // 6))
        with cB:
            offset_y = st.slider("Notch offset Y", -H // 2, H // 2, 0)
        with cC:
            notch_r = st.slider("Notch radius (px)", 1, 30, 6)
        mask_preset = make_notch_mask((H, W), offset_x, offset_y, notch_r)
        explanation_key = "notch"
    elif preset == "Directional strip":
        orientation = st.radio("Strip orientation", ["horizontal", "vertical"], horizontal=True)
        strip_w = st.slider("Strip width (px)", 1, max(H, W) // 2, 8)
        mask_preset = make_directional_strip_mask((H, W), orientation, strip_w)
        explanation_key = "directional"

    recon_p = apply_mask_and_reconstruct(shifted_fft, mask_preset)
    recon_p_disp = normalize_for_display(recon_p)

    coverage = float((mask_preset < 0.5).mean()) * 100.0
    mse = float(np.mean((recon_p_disp - normalize_for_display(working)) ** 2))

    c1, c2, c3 = st.columns(3)
    with c1:
        show_image(mask_preset, caption="Preset mask (white = kept)")
        st.caption(f"Removed area: {coverage:.1f}%")
    with c2:
        show_spectrum(log_spectrum * mask_preset, caption="Masked spectrum")
        st.caption("Only frequencies inside the mask survive.")
    with c3:
        show_image(recon_p_disp, caption="Reconstruction")
        st.caption(f"MSE vs. original (normalised): {mse:.4f}")

    st.info(explain_preset(explanation_key))

    with st.expander("Compare original vs. reconstruction"):
        cL, cR, cD = st.columns(3)
        with cL:
            show_image(working, caption="Original grayscale")
        with cR:
            show_image(recon_p_disp, caption="Reconstruction")
        with cD:
            diff = np.abs(recon_p_disp - normalize_for_display(working))
            show_image(normalize_for_display(diff), caption="|difference|")
