""" Simple script to rename our sample image files for cv testing """

import os

directory_path = "./" 

def rename_images():
    try:
        for i, filename in enumerate(os.listdir(directory_path)):
            if filename.endswith(".jpeg"): # grab jpg files
                old_filepath = os.path.join(directory_path, filename)
                
                # renaming logic
                new_filename = f"tetris_screen_{i}.jpeg" 
                new_filepath = os.path.join(directory_path, new_filename)

                os.rename(old_filepath, new_filepath)
                print(f"Renamed '{filename}' to '{new_filename}'")
    except FileNotFoundError:
        print(f"Error: Directory '{directory_path}' not found.")
    except OSError as e:
        print(f"Error processing files in directory: {e}")

if __name__ == "__main__":
    rename_images()
