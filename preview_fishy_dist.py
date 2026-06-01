import colorsys
import copy
import numpy as np
import sys
sys.path.append("/snap/blender/common/BlenderPython")
from sklearn.datasets import make_blobs
import pandas as pd
import os
import matplotlib.pyplot as plt


"""
This script colors a fishy 3d model.
It generates multiple variations of the fish coloration.
The generated textures are saved as files and a CSV file is generated with the parameters of each texture.

The fish has 5 visual features:
- base color (yellow, with some hue variation, saturation, and value)
- black stripes (with some randomization in width, spacing, and number)
- belly color (blue, with some hue variation, strength, and translation)
- tail color (green, with some hue variation, strength, and translation)
- "rosy cheeks" (red, with some hue variation, strength, and translation)


The fish color palette consists of the following colors:
blue (for belly), yellow (for base), green (for tail), black (for stripes), red (for rosy cheeks)

The following parameters exist and are sampled from a distribution:
- base_color_hue: sampled from a normal distribution
- base_color_sat: sampled from a normal distribution
- base_color_val: sampled from a normal distribution
- stripe_count: 80% chance of 4, 20% chance of 5        <---- cluster feature
- stripe_spacing: sampled from a normal distribution
- stripe_width: sampled from a normal distribution
- stripe_longitudinal_offset: sampled from a normal distribution
- belly_hue: two hue clusters, 1d make_blobs            <---- cluster feature
- belly_strength: sampled from a normal distribution
- belly_translation: sampled from a normal distribution
- tail_hue: two hue clusters, 1d make_blobs             <---- cluster feature
- tail_strength: sampled from a normal distribution
- rosy_cheeks_present: 5% chance of being present       <---- cluster feature
- rosy_cheeks_hue: sampled from a normal distribution, 0 if not present
- rosy_cheeks_strength: sampled from a normal distribution, 0 if not present
- rosy_cheeks_translation_y: sampled from a normal distribution, 0 if not present
- rosy_cheeks_translation_z: sampled from a normal distribution, 0 if not present

"""

# OUTPUT_DIR = "/home/alek/projects/3d_color_package/data/fishy"
OUTPUT_DIR = "/tmp/ml_data/fishy"
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMAGE_SIZE = 2048  # 1024..8192
IMPORT_MESH_PATH = None 

SEED = 49 #47
N_SAMPLES = 250

ADD_NOISE = True
NOISE_AMOUNT = 0.10    # blend weight
NOISE_SCALE = 8.0      # higher = finer noise

# Palette
BLUE = colorsys.rgb_to_hsv(39/255., 70/255., 144/255.)
YELLOW = colorsys.rgb_to_hsv(242/255., 255/255., 73/255.)
GREEN = colorsys.rgb_to_hsv(21/255., 127/255., 31/255.)
BLACK = colorsys.rgb_to_hsv(0., 0., 0.)
RED = colorsys.rgb_to_hsv(255/255., 102/255., 99/255.)

BASE_COLOR_HUE_STD = 0.015
BASE_COLOR_SAT_STD = 0.05
BASE_COLOR_VAL_STD = 0.05

NOMINAL_FIRST_STRIPE_START = (0.1, -0.25, 0.07)
NOMINAL_FIRST_STRIPE_END = (0.0, -0.24, 0.17)

STRIPE_COUNT_PROB_4 = 0.8

NOMINAL_STRIPE_SPACING = 0.08
STRIPE_SPACING_STD = 0.01

NOMINAL_STRIPE_WIDTH = 0.03
STRIPE_WIDTH_STD = 0.005

NOMINAL_STRIPE_LONGITUDINAL_OFFSET = 0.00
STRIPE_LONGITUDINAL_OFFSET_STD = 0.01

# for make_blobs
BELLY_HUE_SHIFT_RANGE = (-0.1, 0.1)
BELLY_HUE_SHIFT_STD = 0.015

BELLY_STRENGTH_STD = 0.05
BELLY_TRANSLATION_STD = 0.05

TAIL_HUE_SHIFT_RANGE = (-0.1, 0.1)
TAIL_HUE_SHIFT_STD = 0.015

TAIL_STRENGTH_STD = 0.05

ROSY_CHEEKS_PROB = 0.05
ROSY_CHEEKS_HUE_STD = 0.015
ROSY_CHEEKS_STRENGTH_STD = 0.05
ROSY_CHEEKS_TRANSLATION_Y_STD = 0.05
ROSY_CHEEKS_TRANSLATION_Z_STD = 0.05

def sample_space(n_samples, random_state=42):
    np.random.seed(random_state)
    base_color_hue = np.random.normal(YELLOW[0], BASE_COLOR_HUE_STD, n_samples) % 1.0
    base_color_sat = np.clip(np.random.normal(0, BASE_COLOR_SAT_STD, n_samples) + YELLOW[1], 0, 1)
    base_color_val = np.clip(np.random.normal(0, BASE_COLOR_VAL_STD, n_samples) + YELLOW[2], 0, 1)
    stripe_count = np.random.choice([4, 5], n_samples, p=[STRIPE_COUNT_PROB_4, 1 - STRIPE_COUNT_PROB_4])
    stripe_spacing = np.random.normal(NOMINAL_STRIPE_SPACING, STRIPE_SPACING_STD, n_samples)
    stripe_width = np.random.normal(NOMINAL_STRIPE_WIDTH, STRIPE_WIDTH_STD, n_samples)
    stripe_longitudinal_offset = np.random.normal(NOMINAL_STRIPE_LONGITUDINAL_OFFSET, STRIPE_LONGITUDINAL_OFFSET_STD, n_samples)
    belly_hue_shift = make_blobs(n_samples=n_samples, n_features=1, centers=2, cluster_std=BELLY_HUE_SHIFT_STD, center_box=BELLY_HUE_SHIFT_RANGE)[0][:, 0]
    belly_hue = (BLUE[0] + belly_hue_shift) % 1.0
    belly_strength = np.clip(np.random.normal(1.0 - BELLY_STRENGTH_STD, BELLY_STRENGTH_STD, n_samples), 0, 1)
    belly_translation = np.random.normal(0, BELLY_TRANSLATION_STD, n_samples)
    tail_hue_shift = make_blobs(n_samples=n_samples, n_features=1, centers=2, cluster_std=TAIL_HUE_SHIFT_STD, center_box=TAIL_HUE_SHIFT_RANGE)[0][:, 0]
    tail_hue = (GREEN[0] + tail_hue_shift) % 1.0
    tail_strength = np.clip(np.random.normal(1.0 - TAIL_STRENGTH_STD, TAIL_STRENGTH_STD, n_samples), 0, 1)
    rosy_cheeks_present = np.random.choice([0, 1], n_samples, p=[1 - ROSY_CHEEKS_PROB, ROSY_CHEEKS_PROB])
    rosy_cheeks_hue = np.random.normal(RED[0], ROSY_CHEEKS_HUE_STD, n_samples) % 1.0
    rosy_cheeks_strength = np.clip(np.random.normal(1.0 - ROSY_CHEEKS_STRENGTH_STD, ROSY_CHEEKS_STRENGTH_STD, n_samples), 0, 1)
    rosy_cheeks_translation_y = np.random.normal(0, ROSY_CHEEKS_TRANSLATION_Y_STD, n_samples)
    rosy_cheeks_translation_z = np.random.normal(0, ROSY_CHEEKS_TRANSLATION_Z_STD, n_samples)
    
    data = [{
        "base_color_hue": base_color_hue[i],
        "base_color_sat": base_color_sat[i],
        "base_color_val": base_color_val[i],
        "stripe_count": stripe_count[i],
        "stripe_spacing": stripe_spacing[i],
        "stripe_width": stripe_width[i],
        "stripe_longitudinal_offset": stripe_longitudinal_offset[i],
        "belly_hue": belly_hue[i],
        "belly_strength": belly_strength[i],
        "belly_translation": belly_translation[i],
        "tail_hue": tail_hue[i],
        "tail_strength": tail_strength[i],
        "rosy_cheeks_present": rosy_cheeks_present[i],
        "rosy_cheeks_hue": rosy_cheeks_hue[i],
        "rosy_cheeks_strength": rosy_cheeks_strength[i],
        "rosy_cheeks_translation_y": rosy_cheeks_translation_y[i],
        "rosy_cheeks_translation_z": rosy_cheeks_translation_z[i],
    } for i in range(n_samples)]
    return data

def generate_stripes(
        stripe_spacing,
        stripe_width,
        stripe_longitudinal_offset,
        stripe_count,
):
    stripes = []
    for i in range(stripe_count):
        stripe = {
            "type": "capsule", 
            "a": NOMINAL_FIRST_STRIPE_START, 
            "b": NOMINAL_FIRST_STRIPE_END,
            "sigma": stripe_width, 
            "color": BLACK, 
            "strength": 1.0
            }
        stripe["b"] = (stripe["b"][0], stripe["b"][1] + i*stripe_spacing + stripe_longitudinal_offset, stripe["b"][2])
        stripe["a"] = (stripe["a"][0], stripe["a"][1] + i*stripe_spacing + stripe_longitudinal_offset, stripe["a"][2])
        stripes.append(stripe)
        mirrored_stripe = copy.deepcopy(stripe)
        mirrored_stripe["a"] = (-stripe["a"][0], stripe["a"][1], stripe["a"][2])
        mirrored_stripe["b"] = (-stripe["b"][0], stripe["b"][1], stripe["b"][2])
        stripes.append(mirrored_stripe)
    return stripes

def generate_base_color(base_color_hue, base_color_sat, base_color_val):
    rgb_color = colorsys.hsv_to_rgb(base_color_hue, base_color_sat, base_color_val)
    return [
        {"type": "sphere", "center": (0.0, 0.00, 0.0), "solid_radius": 2, "sigma": 1, "color": rgb_color, "strength": 1.0},
    ]

def generate_belly(belly_hue, belly_strength, belly_translation):
    rgb_color = colorsys.hsv_to_rgb(belly_hue, BLUE[1], BLUE[2])
    return [
        {"type": "sphere", "center": (0.0, -0.1+belly_translation, -0.55), "solid_radius": 0.5, "sigma": 0.05, "color": rgb_color, "strength": belly_strength},
    ]

def generate_tail(tail_hue, tail_strength):
    rgb_color = colorsys.hsv_to_rgb(tail_hue, GREEN[1], GREEN[2])
    return [
        {"type": "sphere", "center": (0.0, 0.5, 0), "solid_radius": 0.05, "sigma": 0.2, "color": rgb_color, "strength": tail_strength},
    ]
    
def generate_rosy_cheeks( rosy_cheeks_present,rosy_cheeks_hue, rosy_cheeks_strength, rosy_cheeks_translation_y, rosy_cheeks_translation_z):
    rgb_color = colorsys.hsv_to_rgb(rosy_cheeks_hue, RED[1], RED[2])
    if rosy_cheeks_present == 0:
        return []
    return [
        {"type": "sphere", "center": (0.07, -0.37+rosy_cheeks_translation_y, -0.03+rosy_cheeks_translation_z), "solid_radius": 0.02, "sigma": 0.01, "color": rgb_color, "strength": rosy_cheeks_strength},
        {"type": "sphere", "center": (-0.07, -0.37+rosy_cheeks_translation_y, -0.03+rosy_cheeks_translation_z), "solid_radius": 0.02, "sigma": 0.01, "color": rgb_color, "strength": rosy_cheeks_strength},
    ]

def generate_fishy_coloration(
        base_color_hue,
        base_color_sat,
        base_color_val,
        stripe_count,
        stripe_spacing,
        stripe_width,
        stripe_longitudinal_offset,
        belly_hue,
        belly_strength,
        belly_translation,
        tail_hue,
        tail_strength,
        rosy_cheeks_present,
        rosy_cheeks_hue,
        rosy_cheeks_strength,
        rosy_cheeks_translation_y,
        rosy_cheeks_translation_z,
):
    fishy = []
    fishy += generate_base_color(base_color_hue, base_color_sat, base_color_val)
    fishy += generate_belly(belly_hue, belly_strength, belly_translation)
    fishy += generate_tail(tail_hue, tail_strength)
    fishy += generate_rosy_cheeks(rosy_cheeks_present, rosy_cheeks_hue, rosy_cheeks_strength, rosy_cheeks_translation_y, rosy_cheeks_translation_z)
    fishy += generate_stripes(stripe_spacing, stripe_width, stripe_longitudinal_offset, stripe_count)
    return fishy

def main():
    samples = sample_space(N_SAMPLES, random_state=SEED)
    df = pd.DataFrame(samples)
    # df.to_csv(os.path.join(OUTPUT_DIR, "samples.csv"), index=False)

    plt.hist(df["belly_hue"], bins=50, alpha=0.5, label="Belly Hue")
    plt.hist(df["tail_hue"], bins=50, alpha=0.5, label="Tail Hue")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
