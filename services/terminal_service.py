import sys

def print_progress_bar(current: int, total: int, bar_length: int = 40) -> None:
    """
    Displays a progress bar in the terminal to indicate the current progress of a process.
    Args:
        current (int): Current progress (e.g., number of items processed).
        total (int): Total number of items to process.
        bar_length (int, optional): Length of the progress bar. Default is 40.
    """
    percent = float(current) / total
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write(f'\rProgress: [{arrow}{spaces}] {current}/{total}')
    sys.stdout.flush()