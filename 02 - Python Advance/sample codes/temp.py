# import tempfile
import tempfile

# create a temporary file and write some data to it
temp_file= tempfile.TemporaryFile()
temp_file.write(b'This is some temporary data1')
temp_file_name = temp_file.name
print(f'Temporary file created: {temp_file_name}')
# read the data back
temp_file.seek(0)
data = temp_file.read()
print(f'Data read from temporary file: {data.decode()}')

# The file will be deleted when closed
temp_file.close()


# Create a temporary file using a context manager  
with tempfile.TemporaryFile() as temp_file:
    temp_file.write(b'This is some temporary data2')
    temp_file.seek(0)
    data = temp_file.read()
    print(f'Data read from temporary file: {data.decode()}')
# The file will be deleted when the context manager exits


# create a temporary directory using context manager
with tempfile.TemporaryDirectory() as temp_dir:
    print(f'Temporary directory created: {temp_dir}')
    # create a temporary file in the temporary directory
    temp_file = tempfile.NamedTemporaryFile(dir=temp_dir, delete=False)
    temp_file.write(b'This is some temporary data3')
    temp_file_name = temp_file.name
    print(f'Temporary file created in temporary directory: {temp_file_name}')
    # read the data back
    temp_file.seek(0)
    data = temp_file.read()
    print(f'Data read from temporary file: {data.decode()}')
    # close the temporary file
    temp_file.close()
# The temporary directory and its contents will be deleted when the context manager exits

