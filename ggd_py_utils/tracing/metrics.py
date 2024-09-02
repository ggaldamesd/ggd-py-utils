from contextlib import contextmanager

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
    from chime import theme

    theme(name="mario")
    
    from time import time

    start_time: float = time()
    yield
    elapsed_time: float = time() - start_time
    
    from colorama import Fore, Style

    if block_name:
        print(f"{Fore.CYAN}Trace: {block_name} -> {Fore.YELLOW}Time: {Fore.GREEN}{elapsed_time:.4f} seconds{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}Time: {Fore.GREEN}{elapsed_time:.4f} seconds{Style.RESET_ALL}")

    from chime import success, theme
    
    success()