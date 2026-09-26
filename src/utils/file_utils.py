from pathlib import Path


def create_dir(directory):        
    directory.mkdir(parents=True, exist_ok=True)   

def change_file_extension(file_name, extension, output_folder= None):
    curr_extension = Path(file_name).suffix.lower()
    file_name= file_name.replace(curr_extension, extension)          
    if output_folder is None:
        return file_name
    else:
        return Path(output_folder) / file_name    