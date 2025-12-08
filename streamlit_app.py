import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import common_utils
from common_utils import *

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


def main():
    st.title("Conditional Image Generator — choose categories")

    categories, cat_to_idx, idx_to_cat, folder_to_cats, groups = load_metadata()
    device = get_device()

    # Sidebar for options
    st.sidebar.header("Settings")
    with st.sidebar.expander("Model"):
        model_choice = st.selectbox("Prefer checkpoint", ["fm_conditional_checkpoints/best_model.pt", "ddpm_conditional_checkpoints/best_model.pt", "None (use uninitialized model)"], index=0)
        guidance_scale = st.slider("Guidance scale", 0.0, 10.0, 3.0, 0.1)
        steps = st.slider("Sampling steps", 5, 200, 50, 5)
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
                final_img, traj = sample_ode_euler_cfg_local(model, label_vec, steps=steps, guidance_scale=guidance_scale, device=device)
            pil = tensor_to_pil(final_img)
            # Use explicit width to avoid overly large display
            st.image(pil, caption="Generated image", width=int(output_width), use_container_width =False)


if __name__ == "__main__":
    main()
