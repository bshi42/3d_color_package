import os
import shutil
import glob

# Configuration
SOURCE_DIR = "/Users/leyangloh/Downloads/3D_Fish/mussels_31"
TARGET_DIR = "/Users/leyangloh/Downloads/3D_Fish/Group_A"

GROUP_IDS = [
    "UF_IZ_507382",
    "UF_IZ_507470",
    "UF_IZ_507653",
    "UF_IZ_507720",
    "UF_IZ_507756"
]

SUBDIRS = ["landmarks", "models", "textures"]
# Mapping of source subdir names to target subdir names if they differ
# Based on existing structure: LMs, models, textures
# Wait, let's check source dir structure again.
# user said "using the models/ and LMs/ directory" in step 41.
# step 51 ls showed: LMs, models, textures.
# So source has: LMs, models, textures.
# Target should have: landmarks, models, textures (as requested "landmark, texture, and model directory")
# Wait, user said "landmark, texture, and model directory" in singular, likely implies standard naming.
# In step 17, I used "landmarks", "models", "textures". I should stick to that convention.

SRC_MAP = {
    "landmarks": "LMs",
    "models": "models",
    "textures": "textures"
}

def main():
    if not os.path.exists(TARGET_DIR):
        print(f"Creating {TARGET_DIR}")
        os.makedirs(TARGET_DIR)
        
    for sub in ["landmarks", "models", "textures"]:
        target_sub = os.path.join(TARGET_DIR, sub)
        if not os.path.exists(target_sub):
            os.makedirs(target_sub)
            
    # Move files
    for mid in GROUP_IDS:
        print(f"Processing {mid}...")
        
        # Landmarks
        # Source: LMs/ID.mrk.json
        src_lm = os.path.join(SOURCE_DIR, "LMs", f"{mid}.mrk.json")
        dst_lm = os.path.join(TARGET_DIR, "landmarks", f"{mid}.mrk.json")
        
        if os.path.exists(src_lm):
            print(f"Moving {src_lm} -> {dst_lm}")
            shutil.move(src_lm, dst_lm)
        else:
            print(f"Warning: Landmark not found: {src_lm}")
            
        # Models
        # Source: models/ID.obj and potentially .mtl
        # Check for files starting with ID in models/
        # Be careful not to move other files if names overlap (unlikely with full ID)
        model_files = glob.glob(os.path.join(SOURCE_DIR, "models", f"{mid}*"))
        for f in model_files:
            basename = os.path.basename(f)
            dst_f = os.path.join(TARGET_DIR, "models", basename)
            print(f"Moving {f} -> {dst_f}")
            shutil.move(f, dst_f)
            
        # Textures
        # Source: textures/ID* (usually png/tiff)
        texture_files = glob.glob(os.path.join(SOURCE_DIR, "textures", f"{mid}*"))
        for f in texture_files:
            basename = os.path.basename(f)
            dst_f = os.path.join(TARGET_DIR, "textures", basename)
            print(f"Moving {f} -> {dst_f}")
            shutil.move(f, dst_f)

    print("Done.")

if __name__ == "__main__":
    main()
