from pathlib import Path

def rename_images(folder_path, prefix="img"):
    """Rename only image files to img1, img2, etc."""
    folder = Path(folder_path)
    
    # Get all image files and sort them
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(folder.glob(ext))
        image_files.extend(folder.glob(ext.upper()))  # Handle uppercase too
    
    image_files.sort()
    
    # Rename sequentially
    for i, old_file in enumerate(image_files, 1):
        new_name = f"{prefix}{i}{old_file.suffix.lower()}"
        new_path = folder / new_name
        
        if not new_path.exists():
            print(f"Renaming: {old_file.name} → {new_name}")
            old_file.rename(new_path)
        else:
            print(f"Skipping {old_file.name} - {new_name} already exists")

# Usage
rename_images(Path("../../datasets/iuh/widefield/"))
