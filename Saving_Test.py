import os

directory_path = r'C:\Users\The_PowerHouse\OneDrive - Newcastle University\Desktop\CSC8099\TrainedModel'

# Check if directory exists
if os.path.exists(directory_path):
    # Check if directory is writable
    if os.access(directory_path, os.W_OK):
        print("Directory exists and is writable.")
    else:
        print("Directory exists but is not writable.")
else:
    print("Directory does not exist.")