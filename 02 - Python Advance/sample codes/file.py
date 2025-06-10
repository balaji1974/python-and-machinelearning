# This is a sample Python script to demonstrate the use of the os module
import os

# GET the current working directory
current_directory = os.getcwd()

# Print the current working directory
print("Current Directory:", current_directory)

# Create a new directory only if it doesn't exist
if not os.path.exists("new_directory"): 
    # Create a new directory
    os.mkdir("new_directory",)

# List all files and directories in the current directory
files_and_directories = os.listdir(current_directory)

# Print the list of files and directories
print("Files and Directories:", files_and_directories)

import time

# Sleep for 2 seconds
time.sleep(2)

# Rename the new directory
os.rename("new_directory", "renamed_directory")

# List all files and directories again
files_and_directories = os.listdir(current_directory)

# Print the updated list of files and directories
print("Updated Files and Directories:", files_and_directories)

# Remove the renamed directory
os.rmdir("renamed_directory")

# List all files and directories again
files_and_directories = os.listdir(current_directory)

# Print the final list of files and directories
print("Final Files and Directories:", files_and_directories)

