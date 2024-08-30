from os import stat, stat_result

def get_file_size(filename:str) -> tuple[float, str]:
    """
    Reads a file and calculates its size in megabytes (MB).

    Parameters
    ----------
    filename : str
        The path to the file to read.

    Returns
    -------
    tuple
        A tuple containing the file size as a float and as a formatted string.
    """
    file_stats:stat_result = stat(path=filename)
    file_size:float = file_stats.st_size
    file_size /= 1024.0**2
    file_size_str:str = f"{file_size:.2f} MB"
    
    return (file_size, file_size_str)