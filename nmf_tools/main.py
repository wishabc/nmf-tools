from nmf_tools import __version__


def print_help():
    lines = ['-h Print help information.']
    print('\n'.join(lines))

def main():
    print_help()
    print(f'nmf-tools version: {__version__}')
