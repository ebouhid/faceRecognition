import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from tqdm import tqdm

class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, transforms=None, debug=False):
        """
        Args:
            root_dir (string): Directory with all the images, organized in subfolders per celebrity.
            transforms (callable, optional): Optional albumentations transform to be applied
                                             on a sample.
            debug (bool): If True, prints debug information.
        """
        self.root_dir = root_dir
        self.provided_transforms = transforms # Transforms passed from the training script
        self.debug = debug
        self.deterministic_mode = False  # For consistent validation triplets
        self._triplet_cache = {}  # Cache for deterministic triplets

        self.all_image_paths = [] # Stores full path to every image
        self.celebrity_to_paths_map = {} # Maps celebrity_id to a list of their image full paths
        self.path_to_celebrity_id = {} # Maps image full path to its celebrity_id

        # Scan directories and populate image lists and maps
        # Sort celebrity folders and images for reproducibility
        celebrity_folder_names = sorted(os.listdir(self.root_dir))
        if not celebrity_folder_names:
            raise ValueError(f"No celebrity folders found in root_dir: {self.root_dir}")

        for celebrity_id in celebrity_folder_names:
            celebrity_folder_path = os.path.join(self.root_dir, celebrity_id)
            if not os.path.isdir(celebrity_folder_path):
                continue

            image_names_in_folder = sorted(os.listdir(celebrity_folder_path))
            current_celebrity_image_paths = []
            for img_name in image_names_in_folder:
                # Basic check for common image file extensions
                if not (img_name.lower().endswith((".jpg", ".jpeg", ".png"))):
                    if self.debug:
                        print(f"Skipping non-image file: {img_name} in folder {celebrity_id}")
                    continue
                
                full_image_path = os.path.join(celebrity_folder_path, img_name)
                self.all_image_paths.append(full_image_path)
                current_celebrity_image_paths.append(full_image_path)
                self.path_to_celebrity_id[full_image_path] = celebrity_id
            
            if current_celebrity_image_paths:
                self.celebrity_to_paths_map[celebrity_id] = current_celebrity_image_paths
            elif self.debug:
                 print(f"No valid images found for celebrity: {celebrity_id}")


        if not self.all_image_paths:
            raise ValueError(f"No images found in any subdirectories of {self.root_dir}")

        self.available_celebrity_ids = list(self.celebrity_to_paths_map.keys())
        if len(self.available_celebrity_ids) < 2:
            raise ValueError("Triplet loss requires at least two different celebrities with images.")

        # Mandatory augmentations for the positive sample if it's the same image file as the anchor.
        # This ensures the model doesn't see the exact same tensor for anchor and positive.
        self.mandatory_positive_transform = A.Compose([
            A.Rotate(limit=30, p=0.9), # Higher probability to ensure augmentation
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4, brightness_limit=0.3, contrast_limit=0.3),
        ])
    
    def get_id(self, idx):
        """Get person ID for a given index"""
        if idx < 0 or idx >= len(self.all_image_paths):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.all_image_paths)}")
        image_path = self.all_image_paths[idx]
        return image_path.split(os.sep)[-2]

    def set_deterministic_mode(self, deterministic=True):
        """Enable/disable deterministic triplet selection for validation"""
        self.deterministic_mode = deterministic
        if deterministic:
            # Pre-compute deterministic triplets for validation
            self._precompute_triplets()

    def _precompute_triplets(self):
        """Pre-compute deterministic positive/negative pairs for each anchor"""
        import hashlib
        
        for idx, anchor_path in enumerate(tqdm(self.all_image_paths, desc="Precomputing triplets")):
            anchor_celebrity_id = self.path_to_celebrity_id[anchor_path]
            
            # Deterministic positive selection
            positive_options = self.celebrity_to_paths_map[anchor_celebrity_id]
            if len(positive_options) == 1:
                positive_path = anchor_path
            else:
                # Use hash for deterministic selection
                seed_str = f"{anchor_path}_positive"
                hash_obj = hashlib.md5(seed_str.encode())
                hash_int = int(hash_obj.hexdigest(), 16)
                available_positives = [p for p in positive_options if p != anchor_path]
                positive_path = available_positives[hash_int % len(available_positives)]
            
            # Deterministic negative selection
            available_negatives = [cid for cid in self.available_celebrity_ids if cid != anchor_celebrity_id]
            if not available_negatives:
                # Skip if no negative celebrities available
                continue
                
            seed_str = f"{anchor_path}_negative"
            hash_obj = hashlib.md5(seed_str.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            negative_celebrity_id = available_negatives[hash_int % len(available_negatives)]
            
            # Select negative image deterministically
            negative_options = self.celebrity_to_paths_map[negative_celebrity_id]
            seed_str = f"{anchor_path}_negative_img"
            hash_obj = hashlib.md5(seed_str.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            negative_path = negative_options[hash_int % len(negative_options)]
            
            self._triplet_cache[idx] = (positive_path, negative_path)

    def __len__(self):
        # The length of the dataset is the total number of images, each can be an anchor.
        return len(self.all_image_paths)

    def __getitem__(self, idx):
        anchor_image_path = self.all_image_paths[idx]
        anchor_celebrity_id = self.path_to_celebrity_id[anchor_image_path]

        if self.deterministic_mode and idx in self._triplet_cache:
            # Use pre-computed deterministic triplets for validation
            positive_image_path, negative_image_path = self._triplet_cache[idx]
        else:
            # Random triplet selection for training
            # Select positive image
            positive_options = self.celebrity_to_paths_map[anchor_celebrity_id]
            if len(positive_options) == 1:
                # Only one image for this celebrity, so positive is same as anchor
                positive_image_path = anchor_image_path
            else:
                # Randomly choose a different image of the same celebrity
                positive_image_path = random.choice(positive_options)
                while positive_image_path == anchor_image_path:
                    positive_image_path = random.choice(positive_options)
            
            # Select negative image (from a different celebrity)
            negative_celebrity_id = random.choice(self.available_celebrity_ids)
            while negative_celebrity_id == anchor_celebrity_id:
                negative_celebrity_id = random.choice(self.available_celebrity_ids)
            
            negative_image_path = random.choice(self.celebrity_to_paths_map[negative_celebrity_id])

        try:
            # Load images as PIL, convert to RGB
            anchor_pil = Image.open(anchor_image_path).convert('RGB')
            positive_pil = Image.open(positive_image_path).convert('RGB')
            negative_pil = Image.open(negative_image_path).convert('RGB')
        except FileNotFoundError as e:
            print(f"ERROR: Image not found: {e}. Skipping this triplet by returning a random one.")
            # Fallback to prevent Dataloader crashing. In a real scenario, clean your dataset.
            return self.__getitem__(random.randint(0, len(self.all_image_paths) - 1))


        # Convert PIL images to NumPy arrays
        anchor_np = np.array(anchor_pil)
        positive_np = np.array(positive_pil)
        negative_np = np.array(negative_pil)

        # Apply main transforms (e.g., resize, normalization) if provided
        # These transforms are expected to output NumPy arrays
        if self.provided_transforms:
            anchor_transformed_np = self.provided_transforms(image=anchor_np)['image']
            positive_transformed_np = self.provided_transforms(image=positive_np)['image']
            negative_transformed_np = self.provided_transforms(image=negative_np)['image']
        else:
            anchor_transformed_np = anchor_np
            positive_transformed_np = positive_np
            negative_transformed_np = negative_np
            

        if self.debug and idx % 100 == 0 : # Print occasionally for debugging
            print(f"\n--- Debug sample {idx} ---")
            print(f"Anchor: {anchor_image_path} (Celeb: {anchor_celebrity_id})")
            print(f"Positive: {positive_image_path} (Celeb: {anchor_celebrity_id})")
            print(f"Negative: {negative_image_path} (Celeb: {negative_celebrity_id})")
            print(f"Shape ANP (after main transform, before ToTensor): {anchor_transformed_np.shape}, {positive_transformed_np.shape}, {negative_transformed_np.shape}")

        return anchor_transformed_np, positive_transformed_np, negative_transformed_np

    def filter_by_celebrities(self, allowed_celebrities):
        """Filter dataset to only include specified celebrities"""
        allowed_celebrities = set(allowed_celebrities)
        
        # Filter all_image_paths to only include allowed celebrities
        self.all_image_paths = [
            path for path in self.all_image_paths 
            if self.path_to_celebrity_id[path] in allowed_celebrities
        ]
        
        # Filter celebrity_to_paths_map
        self.celebrity_to_paths_map = {
            celebrity_id: paths for celebrity_id, paths in self.celebrity_to_paths_map.items()
            if celebrity_id in allowed_celebrities
        }
        
        # Update available_celebrity_ids
        self.available_celebrity_ids = list(self.celebrity_to_paths_map.keys())
        
        # Clear cache if in deterministic mode
        if self.deterministic_mode:
            self._triplet_cache = {}
            self._precompute_triplets()
