def abbreviate_large_number(number: float) -> str:
    """
    Abbreviate a large number into a shorter string using K, M, B, T suffixes.

    Example:
        >>> abbreviate_large_number(1234)
        '1K'
        >>> abbreviate_large_number(1234567)
        '1M'
        >>> abbreviate_large_number(1234567890)
        '1B'
        >>> abbreviate_large_number(1234567890123)
        '1T'

    :param number: The number to abbreviate
    :return: The abbreviated string
    """
    suffixes: list[str] = ['', 'K', 'M', 'B', 'T']
    index = 0

    while number >= 1000 and index < len(suffixes) - 1:
        number /= 1000.0
        index += 1

    return f'{number:.0f}{suffixes[index]}'