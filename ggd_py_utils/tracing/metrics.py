from contextlib import contextmanager

def human_friendly_time(elapsed_time:float) -> str:
    """
    Convert a time in seconds to a human-friendly string.

    Parameters
    ----------
    elapsed_time : float
        The time in seconds to convert.

    Returns
    -------
    str
        A string representing the time in a human-friendly format,
        e.g. "1h 30m 45.67s" for 5445.67 seconds.

    """
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    output: str = None

    if hours > 0:
        output = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif minutes > 0:
        output = f"{int(minutes)}m {seconds:.2f}s"
    else:
        output =f"{seconds:.2f}s"

    return output

@contextmanager
def time_block(block_name:str=None):
    """
    Context manager to measure the execution time of a code block.

    This context manager will print the execution time of the block of code
    inside the with statement in seconds with four decimal places.

    Parameters
    ----------
    block_name : str, optional
        The name of the block to be printed before the execution time.

        
    """
    from time import time

    start_time: float = time()
    yield
    elapsed_time: float = time() - start_time
    
    elapsed_time_str:str = human_friendly_time(elapsed_time=elapsed_time)
    
    # from colorama import Fore, Style

    if block_name:
        # print(f"{Fore.CYAN}Trace: {block_name} -> {Fore.YELLOW}Took: {Fore.GREEN}{elapsed_time_str}{Style.RESET_ALL}")
        print(f"Trace: {block_name} -> Took: {elapsed_time_str}")
    else:
        # print(f"{Fore.YELLOW}Took: {Fore.GREEN}{elapsed_time_str}{Style.RESET_ALL}")
        print(f"Took: {elapsed_time_str}")

    from chime import success, theme
    
    theme(name="mario")
    
    success()