import os
from tqdm import tqdm
import SimpleITK as sitk

def create_directories(output_base_dir, subdirs):
    """Create directory structure for output data."""
    for subdir in subdirs:
        dir_path = os.path.join(output_base_dir, subdir)
        os.makedirs(dir_path, exist_ok=True)

def extract_slices(input_dir, output_dir):
    """Extract slices from 3D NIfTI images and save as separate files."""
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files):
            if file.endswith(".nii.gz"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                # Load 3D NIfTI file
                img = sitk.ReadImage(file_path)
                data = sitk.GetArrayFromImage(img)

                # Extract and save each slice
                for slice_idx in range(data.shape[0]):
                    slice_data = data[slice_idx, :, :]
                    slice_img = sitk.GetImageFromArray(slice_data)
                    # slice_img.CopyInformation(img)

                    # Save the slice as a new NIfTI file
                    slice_filename = f"{file[:-7]}_{slice_idx:02d}.nii.gz"
                    slice_path = os.path.join(output_subdir, slice_filename)
                    sitk.WriteImage(slice_img, slice_path)

                    print(f"Saved slice: {slice_path}")

if __name__ == "__main__":
    # Define input and output directory structure
    input_base_dir = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/data/Promise_nii_preprocessed"
    output_base_dir = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/sam_data/Promise12"

    subdirs = [
        "train/images",
        "train/labels",
        "valid/images",
        "valid/labels",
        "test/images",
        "test/labels",
    ]

    # Create output directory structure
    create_directories(output_base_dir, subdirs)

    # Process images and labels
    extract_slices(os.path.join(input_base_dir, "train/images"), os.path.join(output_base_dir, "train/images"))
    extract_slices(os.path.join(input_base_dir, "train/labels"), os.path.join(output_base_dir, "train/labels"))
    extract_slices(os.path.join(input_base_dir, "valid/images"), os.path.join(output_base_dir, "valid/images"))
    extract_slices(os.path.join(input_base_dir, "valid/labels"), os.path.join(output_base_dir, "valid/labels"))
    extract_slices(os.path.join(input_base_dir, "test/images"), os.path.join(output_base_dir, "test/images"))
    extract_slices(os.path.join(input_base_dir, "test/labels"), os.path.join(output_base_dir, "test/labels"))
