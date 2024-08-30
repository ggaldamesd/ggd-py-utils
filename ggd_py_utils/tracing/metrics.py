from contextlib import contextmanager
from time import time
from colorama import Fore, Style
from chime import success, theme

theme(name="mario")

@contextmanager
def time_block(block_name:str=""):
    """
    Context manager to measure the execution time of a code block.

    This context manager will print the execution time of the block of code
    inside the with statement in seconds with four decimal places.

    Parameters
    ----------
    block_name : str, optional
        The name of the block to be printed before the execution time.

    """
    
    start_time: float = time()
    yield
    elapsed_time: float = time() - start_time
    
    if block_name:
        print(f"{Fore.CYAN}Trace: {block_name} -> {Fore.YELLOW}Time: {Fore.GREEN}{elapsed_time:.4f} seconds{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}Time: {Fore.GREEN}{elapsed_time:.4f} seconds{Style.RESET_ALL}")

    success()