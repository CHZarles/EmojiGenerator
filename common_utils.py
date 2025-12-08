import random
import torch
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
from torch import nn
import math
import torch.nn.functional as F

# Constants
IMG_SIZE = 64
BATCH_SIZE = 128
SEED = 7402

def setup_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset Utils ---

def load_labels_and_mappings(labels_path='EMOJI_FACE/labels.json'):
    """
    Parse labels.json to construct a mapping from numeric folder id -> multiple category labels.

    - Ignores emoji symbols, focuses only on numeric ids.
    - Builds `categories` as flattened "category:value" strings (e.g., "eye_shape:smiling_curve").
    - Returns mapping from folder id (string) -> list of category indices (`folder_to_cats`).
    """
    with open(labels_path, 'r') as f:
        raw = json.load(f)

    # Build the complete set of (category, value) labels
    flat_labels = []  # list of "category:value" strings
    # Temporary structure to collect id -> set of flat label names
    id_to_labels = {}

    for category_name, value_dict in raw.items():
        if not isinstance(value_dict, dict):
            # Skip unexpected format
            continue
        for value_name, items in value_dict.items():
            # We ignore "default" strings since they don't list explicit ids
            if isinstance(items, str) and items == "default":
                # Still register the value as a possible label, but without explicit ids
                flat_label = f"{category_name}:{value_name}"
                if flat_label not in flat_labels:
                    flat_labels.append(flat_label)
                continue

            # Items are expected to be lists of [id, emoji]; ignore emoji
            flat_label = f"{category_name}:{value_name}"
            if flat_label not in flat_labels:
                flat_labels.append(flat_label)

            if isinstance(items, list):
                for entry in items:
                    if not isinstance(entry, (list, tuple)) or len(entry) < 1:
                        continue
                    num_id = entry[0]
                    fid_str = str(num_id)
                    if fid_str not in id_to_labels:
                        id_to_labels[fid_str] = set()
                    id_to_labels[fid_str].add(flat_label)

    # Create indices for categories
    categories = sorted(flat_labels)
    cat_to_idx = {name: i for i, name in enumerate(categories)}
    idx_to_cat = {i: name for i, name in enumerate(categories)}

    # Map folder id -> list of category indices
    folder_to_cats = {}
    for fid_str, label_names in id_to_labels.items():
        folder_to_cats[fid_str] = [cat_to_idx[name] for name in label_names if name in cat_to_idx]

    print(f"Parsed {len(categories)} category labels and {len(folder_to_cats)} ids from {labels_path}")
    return categories, cat_to_idx, idx_to_cat, folder_to_cats

class ConditionalImageDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, folder_to_cats=None, num_categories=None):
        super().__init__(root, transform=transform)
        self.folder_to_cats = folder_to_cats or {}
        self.num_categories = int(num_categories) if num_categories is not None else 0
        
    def __getitem__(self, index):
        path, _ = self.samples[index]
        folder_name = os.path.basename(os.path.dirname(path))
        
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        
        # Build multi-hot label vector for all categories this id belongs to
        label_vec = torch.zeros(self.num_categories, dtype=torch.float32)
        valid_cats = self.folder_to_cats.get(folder_name, [])
        for ci in valid_cats:
            if 0 <= ci < self.num_categories:
                label_vec[ci] = 1.0
        
        return img, label_vec

def load_conditional_dataset(root_dir="./EMOJI_FACE", folder_to_cats=None, num_categories=None):
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1),  # [-1,1]
    ])

    dataset = ConditionalImageDataset(
        root=root_dir, 
        transform=data_transforms,
        folder_to_cats=folder_to_cats,
        num_categories=num_categories
    )
    
    print(f"Loaded {len(dataset)} images from {root_dir}")
    return dataset

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.clamp(0, 1)),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))
    plt.axis("off")

# --- Model Utils ---

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class AdaGroupNorm(nn.Module):
    def __init__(self, time_emb_dim, num_channels, num_groups=32):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels)
        self.time_proj = nn.Linear(time_emb_dim, num_channels * 2)

    def forward(self, x, time_emb):
        x_norm = self.group_norm(x)
        style = self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        scale, shift = style.chunk(2, dim=1)
        return x_norm * (1 + scale) + shift

class ResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.adagn1 = AdaGroupNorm(time_emb_dim, out_ch, num_groups=8)
        self.relu = nn.SiLU()

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.adagn2 = AdaGroupNorm(time_emb_dim, out_ch, num_groups=8)
        self.skip_connection = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.adagn1(h, t)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.adagn2(h, t)
        h = self.relu(h)
        return h + self.skip_connection(x)

class ConditionalUnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512)
        up_channels = (512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 128

        # Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Class Embedding: one embedding per flattened category label
        # Supports multi-label by summing embeddings weighted by a multi-hot vector
        self.class_emb = nn.Embedding(num_classes, time_emb_dim)

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList()
        for i in range(len(down_channels)-1):
            self.downs.append(ResnetBlock(down_channels[i], down_channels[i], time_emb_dim))
            self.downs.append(nn.Conv2d(down_channels[i], down_channels[i+1], 4, 2, 1))

        self.ups = nn.ModuleList()
        for i in range(len(up_channels)-1):
            self.ups.append(nn.ConvTranspose2d(up_channels[i], up_channels[i+1], 4, 2, 1))
            self.ups.append(ResnetBlock(up_channels[i+1] * 2, up_channels[i+1], time_emb_dim))

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep, label_vec):
        # 1. Time Embedding
        t_emb = self.time_mlp(timestep)
        
        # 2. Class Embedding (multi-label):
        # label_vec is expected shape [B, num_classes] multi-hot or soft weights
        # Compute weighted sum of embeddings: [B, D]
        if label_vec.dim() == 1:
            label_vec = label_vec.unsqueeze(0)
        # 相当于查表,然后加权求和  
        c_emb = torch.matmul(label_vec.float(), self.class_emb.weight)
        
        # 3. Combine (Add)
        emb = t_emb + c_emb
        
        x = self.conv0(x)
        
        residual_inputs = []
        
        for i in range(0, len(self.downs), 2):
            resnet = self.downs[i]
            downsample = self.downs[i+1]
            x = resnet(x, emb)
            residual_inputs.append(x)
            x = downsample(x)
            
        for i in range(0, len(self.ups), 2):
            upsample = self.ups[i]
            resnet = self.ups[i+1]
            residual_x = residual_inputs.pop()
            x = upsample(x)
            if x.shape != residual_x.shape:
                x = F.interpolate(x, size=residual_x.shape[2:], mode='nearest')
            x = torch.cat((x, residual_x), dim=1)
            x = resnet(x, emb)
            
        return self.output(x)
if __name__ == "__main__":
    setup_seed()
    device = get_device()
    print(f"Using device: {device}")

    # Example usage
    categories, cat_to_idx, idx_to_cat, folder_to_cats = load_labels_and_mappings(labels_path='EMOJI_FACE/labels.json')
    # print(f"Categories: {categories}")
    #print(f"Category to Index: {cat_to_idx}")
    # print(f"Index to Category: {idx_to_cat}")
    print(f"Folder to Categories: {folder_to_cats}")
    # print(len(categories))
    exit()
    dataset = load_conditional_dataset(root_dir="./EMOJI_FACE", folder_to_cats=folder_to_cats, num_categories=len(categories))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Fetch a batch
    images, labels = next(iter(dataloader))
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")

    # Show an example image
    # show_tensor_image(images[0])

    # Initialize model
    model = ConditionalUnet(num_classes=len(categories)).to(device)
    print(model)