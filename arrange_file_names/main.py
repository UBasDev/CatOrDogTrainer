import os

def rename_images(folder_path):
    # List of supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    
    # Get all files in the specified directory
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Filter for image files based on extensions
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
    
    # Rename the image files
    for index, filename in enumerate(image_files, start=1):
        # Create new filename
        new_name = f"{index}.jpg"
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_name)
        
        if os.path.exists(new_file_path):
            print(f"Cannot rename '{filename}' to '{new_name}' because it already exists.")
            continue  # Skip to the next file
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed '{filename}' to '{new_name}'")

# Specify the path to the folder containing the image files
folder_path = 'D:\\_personal_demos_\\_python_projects_\\MetinStoneDetector\\train\\cats'  # Replace with your folder path

# Call the function to rename images
rename_images(folder_path)