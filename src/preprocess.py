import cv2
import os
import numpy as np

def deskew_image(img):
    """
    Corrects image alignment angles dynamically by targeting dark text foreground pixels
    instead of the bright background canvas.
    """
    # Since background is light (~255) and text is dark (~0), we target pixels below 
    # a dark threshold to isolate true text point coordinates.
    coords = np.column_stack(np.where(img < 100))
    if len(coords) == 0:
        return img
        
    # Get the minimum bounding rectangle around the isolated text coordinate clusters
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    (w, h) = rect[1]

    # Modern OpenCV normalization tracking logic adjustments
    if w < h:
        angle = angle - 90
    else:
        angle = angle

    # Keep rotations minimal (clamp down massive flips)
    if angle < -45:
        angle = +(90 + angle)
    else:
        angle = -angle

    # Limit maximum rotation correction to 15 degrees to avoid accidental sideways flips on square images
    if abs(angle) > 15:
        angle = 0.0

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Use INTER_CUBIC for clean rotation reconstruction, filling borders with pristine white
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )
    return rotated

def resize_image(img, width=1500):
    """
    Upscales tiny thermal fonts cleanly using Lanczos neighborhood calculations
    to avoid character fragmentation.
    """
    h, w = img.shape[:2]
    ratio = width / w
    dim = (width, int(h * ratio))
    
    # Swapped INTER_LINEAR out for INTER_LANCZOS4 to create crisper anti-aliased font edges
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_LANCZOS4)
    return resized

def preprocess_image(input_path, output_path, final_width=1500):
    """
    Advanced Text Preservation Pipeline.
    Stabilizes font contours and structures data arrays before OCR execution.
    """
    # 1. Read Image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Image not found at path: {input_path}")
    
    # 2. Convert to Grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Dynamic Upscaling (Blows up font boundaries using Lanczos)
    if final_width:
        gray_img = resize_image(gray_img, width=final_width)

    # 4. CLAHE Optimization Layer (Executed first to balance image illumination evenly)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    balanced_contrast = clahe.apply(gray_img)

    # 5. Sharpening Filter (Crisps up edge gradients after balancing lighting)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(balanced_contrast, -1, kernel)

    # 6. Morphological Closing (Connects broken pixels and heals fading/dotted text strokes)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    text_healed = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, morph_kernel)

    # 7. Deskew Image smoothly using corrected foreground mapping
    deskewed_img = deskew_image(text_healed)

    # 8. Frame Padding Layer (Adds a clean 40px white cushion to prevent Tesseract margin clipping)
    final_processed_img = cv2.copyMakeBorder(
        deskewed_img, 40, 40, 40, 40, 
        borderType=cv2.BORDER_CONSTANT, 
        value=255
    )

    # Save the polished high-contrast grayscale image matrix
    cv2.imwrite(output_path, final_processed_img)
    return output_path

def batch_preprocess(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)

            try:
                preprocess_image(input_path, output_path)
                print(f"Processed: {file}")
            except Exception as e:
                print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    input_folder = "data/raw_images/train_15"
    output_folder = "data/processed_data/processed_train_15"

    print("="*60)
    print("Initiating Enhanced Computer Vision Preprocessing Matrix...")
    print("="*60)
    batch_preprocess(input_folder, output_folder)
    print("="*60)
    print("SUCCESS: Image enhancement complete.")
    print("="*60)