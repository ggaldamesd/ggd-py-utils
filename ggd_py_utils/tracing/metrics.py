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
    hours_int:int = int(hours)
    
    minutes, seconds = divmod(rem, 60)
    minutes_int:int = int(minutes)
    seconds_str:str = "{:.2f}".format(seconds)
    
    output: str = None

    if hours > 0:
        output = "{}h {}m {}s".format(hours_int, minutes_int, seconds_str)
    elif minutes > 0:
        output = "{}m {}s".format(minutes_int, seconds_str)
    else:
        output ="{}s".format(seconds_str)

    return output

@contextmanager
def time_block(block_name:str=None, should_beep:bool=False):
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
    
    from colorama import Fore, Style

    if block_name:
        elapsed_time_output:str = "{}Trace: {} -> {}Took: {}{}{}".format(Fore.CYAN, block_name, Fore.YELLOW, Fore.GREEN, elapsed_time_str, Style.RESET_ALL)
        print(elapsed_time_output)
    else:
        elapsed_time_output:str = "{}Took: {}{}{}".format(Fore.YELLOW, Fore.GREEN, elapsed_time_str, Style.RESET_ALL)
        print(elapsed_time_output)

    if should_beep:
        from chime import success, theme
        
        theme(name="mario")
        
        success()