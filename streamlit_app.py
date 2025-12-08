import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import common_utils
from common_utils import *
# sys not needed

# --- DDPM schedule & helpers (copied from ddpm_conditional.ipynb) ---
from torch.nn import functional as F

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """Return the appropriate schedule value indexed by t for a batch.

    Matches the `get_index_from_list` implementation in the ddpm notebook.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# Default DDPM schedule length (matches ddpm_conditional notebook)
T = 500
betas = linear_beta_schedule(timesteps=T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

# Keep variable for UI slider default and compatibility
DDPM_T = T


@torch.no_grad()
def sample_timestep_cfg_local(model, x, t, label_vec, guidance_scale=3.0):
    """Single DDPM reverse step with classifier-free guidance (local helper).
    This mirrors the implementation in `ddpm_conditional.ipynb`.
    t must be a long tensor indexing the schedule.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    pred_noise_uncond = model(x, t, torch.zeros_like(label_vec))
    pred_noise_cond = model(x, t, label_vec)
    pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)

    model_mean = sqrt_recip_alphas_t * (x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if int(t.item()) == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_final_ddpm(model, label_vec, steps=None, guidance_scale=3.0, device="cpu"):
    """Simple reverse-time sampler using the full DDPM schedule.

    This mirrors `sample_final_ddpm` in `ddpm_conditional.ipynb` and does not
    take advantage of timestep subsampling; it runs through the original T steps
    in descending order.
    """
    if steps is None:
        steps = T
    steps = min(steps, T)
    x = torch.randn((1, 3, IMG_SIZE, IMG_SIZE), device=device)
    for i in range(steps - 1, -1, -1):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        x = sample_timestep_cfg_local(model, x, t, label_vec, guidance_scale=guidance_scale)
        x = torch.clamp(x, -1.0, 1.0)
    return x
st.set_page_config(page_title="Conditional Generator", layout="wide")

@st.cache_resource
def load_metadata(labels_path="EMOJI_FACE/labels.json"):
    categories, cat_to_idx, idx_to_cat, folder_to_cats = load_labels_and_mappings(labels_path=labels_path)
    # Build grouping by top-level category name
    groups = {}
    for idx, flat in enumerate(categories):
        if ":" in flat:
            grp, val = flat.split(":", 1)
        else:
            grp, val = flat, flat
        groups.setdefault(grp, []).append((idx, val, flat))
    return categories, cat_to_idx, idx_to_cat, folder_to_cats, groups


@st.cache_resource
def load_conditional_model(num_categories, checkpoint_paths=None, device="cpu"):
    model = ConditionalUnet(num_classes=num_categories)
    model.to(device)
    loaded = False
    if checkpoint_paths is None:
        checkpoint_paths = [
            "fm_conditional_checkpoints/best_model.pt",
            "ddpm_conditional_checkpoints/best_model.pt",
            "ddpm_checkpoints/best_model.pt",
        ]
    for p in checkpoint_paths:
        if os.path.exists(p):
            try:
                ckpt = torch.load(p, map_location=device)
                if 'model_state_dict' in ckpt:
                    model.load_state_dict(ckpt['model_state_dict'])
                else:
                    model.load_state_dict(ckpt)
                loaded = True
                loaded_path = p
                break
            except Exception as e:
                print(f"Failed loading {p}: {e}")
    if not loaded:
        loaded_path = None
    model.eval()
    return model, loaded_path


def build_label_vec_from_selected(selected_flat_names, categories, device):
    vec = torch.zeros((1, len(categories)), dtype=torch.float32, device=device)
    name_to_idx = {name: i for i, name in enumerate(categories)}
    for name in selected_flat_names:
        idx = name_to_idx.get(name)
        if idx is not None:
            vec[0, idx] = 1.0
    return vec


@torch.no_grad()
def sample_ode_euler_cfg_local(model, label_vec, steps=30, guidance_scale=3.0, device="cpu"):
    img_size = IMG_SIZE
    x = torch.randn((1, 3, img_size, img_size), device=device)
    dt = 1.0 / steps
    traj = [x.clone()]
    for i in range(steps):
        t_val = i / steps
        t = torch.full((1,), t_val, device=device, dtype=torch.float32)
        v_uncond = model(x, t * 1000, torch.zeros_like(label_vec))
        v_cond = model(x, t * 1000, label_vec)
        v = v_uncond + guidance_scale * (v_cond - v_uncond)
        x = x + v * dt
        if i % max(steps // 10, 1) == 0:
            traj.append(x.clone())
    return x.clamp(-1, 1), traj


def tensor_to_pil(img_tensor):
    t = img_tensor.detach().cpu().squeeze(0)
    t = (t + 1.0) / 2.0
    t = t.clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


@torch.no_grad()
def sample_ddpm_cfg(model, label_vec, steps=None, guidance_scale=3.0, device="cpu"):
    """DDPM reverse-time sampling with classifier-free guidance (CFG).

        Notes:
        - Uses the forward-noising schedule defined in-file (mirrors `ddpm_conditional.ipynb`)
            so the maximum number of steps (`DDPM_T`) and the beta schedule are consistent
            with the training used in the ddpm code (T=500 by default).
    - If `steps` is < `DDPM_T` we sample timesteps evenly spaced across [0, DDPM_T-1].
    """
    img_size = IMG_SIZE
    x = torch.randn((1, 3, img_size, img_size), device=device)

    # Default to the original training T if not provided
    if steps is None:
        steps = int(DDPM_T or 300)
    steps = int(max(1, steps))
    # Timesteps: create an (descending) list of indices into the original schedule
    orig_T = int(DDPM_T or 300)
    # Ensure we hit the full schedule if requested
    if steps >= orig_T:
        t_idxs = np.arange(orig_T - 1, -1, -1)
    else:
        # Sample evenly-spaced timesteps across the original schedule (descending)
        t_idxs = np.linspace(orig_T - 1, 0, steps).astype(int)

    for t_idx in t_idxs:
        t_index = torch.full((1,), int(t_idx), device=device, dtype=torch.long)
        t = t_index  # pass as long to be consistent with ddpm implementation

        x = sample_timestep_cfg_local(model, x, t_index, label_vec, guidance_scale)
    return x.clamp(-1, 1)


def main():
    st.title("Conditional Image Generator — choose categories")

    categories, cat_to_idx, idx_to_cat, folder_to_cats, groups = load_metadata()
    device = get_device()

    # Sidebar for options
    st.sidebar.header("Settings")
    with st.sidebar.expander("Model"):
        model_choice = st.selectbox("Prefer checkpoint", ["fm_conditional_checkpoints/best_model.pt", "ddpm_conditional_checkpoints/best_model.pt", "None (use uninitialized model)"], index=0)
        sampler_choice = st.selectbox("Sampler", ["ODE Euler (Flow Matching)", "DDPM (discrete)"])
        guidance_scale = st.slider("Guidance scale", 0.0, 10.0, 3.0, 0.1)
        # Dynamically set steps slider range depending on sampler choice
        if sampler_choice == "DDPM (discrete)":
            max_steps = int(DDPM_T or 300)
            steps = st.slider("DDPM sampling steps", 1, max_steps, min(300, max_steps), step=1)
        else:
            steps = st.slider("Sampling steps (ODE)", 5, 200, 50, 5)
        output_width = st.slider("Output width (px)", 128, 1024, 384, step=1)

    # Grouped category selection
    st.sidebar.header("Select labels")
    selected_flat = []
    for grp, vals in groups.items():
        with st.sidebar.expander(grp, expanded=False):
            for idx, val, flat in vals:
                if st.checkbox(f"{val}", key=f"{grp}:{val}"):
                    selected_flat.append(flat)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Selected flattened labels")
        if selected_flat:
            for s in selected_flat:
                st.write(s)
        else:
            st.write("(none)")

    # Load model
    model, loaded_path = load_conditional_model(len(categories), checkpoint_paths=[model_choice] if model_choice != "None (use uninitialized model)" else None, device=device)
    if loaded_path is None:
        st.warning("No checkpoint loaded; model is uninitialized (random weights). You can still sample but results will be random.")
    else:
        st.success(f"Loaded checkpoint: {loaded_path}")

    if st.button("Generate"):
        if not selected_flat:
            st.error("Please select at least one label value to condition on.")
        else:
            label_vec = build_label_vec_from_selected(selected_flat, categories, device)
            with st.spinner("Generating..."):
                if sampler_choice == "DDPM (discrete)":
                    final_img = sample_ddpm_cfg(model, label_vec, steps=steps, guidance_scale=guidance_scale, device=device)
                    traj = None
                else:
                    final_img, traj = sample_ode_euler_cfg_local(model, label_vec, steps=steps, guidance_scale=guidance_scale, device=device)
            pil = tensor_to_pil(final_img)
            # Use explicit width to avoid overly large display
            st.image(pil, caption="Generated image", width=int(output_width), use_container_width =False)


if __name__ == "__main__":
    main()
