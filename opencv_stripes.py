import cv2
import numpy as np


def extract_dark_stripes(
    image_path: str,
    gray_threshold: int = 115,
    bstar_threshold: int = 140,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract dark stripes from a clam shell image.

    Returns (stripe_mask, overlay) where stripe_mask is a binary mask of the
    dark stripe regions, and overlay is the original image with stripes highlighted.

    Uses two conditions to separate dark cool-gray stripes from warm golden areas:
      gray_threshold:  pixels darker than this qualify (clam gray range is ~46-215)
      bstar_threshold: pixels with LAB b* below this qualify (b* < 128 = blue/cool,
                       > 128 = yellow/warm; golden tan areas score ~140-162)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Mask out the white background (all channels near 255).
    bg_mask = np.all(image > 230, axis=2)
    clam_mask = (~bg_mask).astype(np.uint8) * 255

    # Light erosion to drop anti-aliased edge fringe without eating into the shell.
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clam_mask = cv2.erode(clam_mask, k3, iterations=1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    b_star = lab[:, :, 2]  # yellow-blue axis: low = cool/gray, high = warm/golden

    dark = (gray < gray_threshold).astype(np.uint8) * 255
    cool = (b_star < bstar_threshold).astype(np.uint8) * 255
    stripe_mask = cv2.bitwise_and(dark, cool)
    stripe_mask = cv2.bitwise_and(stripe_mask, clam_mask)

    # Open removes speckle; close fills small gaps within bands.
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    stripe_mask = cv2.morphologyEx(stripe_mask, cv2.MORPH_OPEN,  k3, iterations=1)
    stripe_mask = cv2.morphologyEx(stripe_mask, cv2.MORPH_CLOSE, k5, iterations=2)

    overlay = image.copy()
    overlay[stripe_mask == 255] = [255, 50, 50]  # BGR: blue highlight

    return stripe_mask, overlay


if __name__ == "__main__":
    mask, overlay = extract_dark_stripes("app.png")

    cv2.imshow("Dark Stripe Mask", mask)
    cv2.imshow("Stripe Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()