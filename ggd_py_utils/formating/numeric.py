def abbreviate_large_number(number: float) -> str:
    suffixes: list[str] = ['', 'K', 'M', 'B', 'T']
    index = 0

    while number >= 1000 and index < len(suffixes) - 1:
        number /= 1000.0
        index += 1

    return f'{number:.0f}{suffixes[index]}'