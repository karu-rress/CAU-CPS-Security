import argparse

parser = argparse.ArgumentParser(description="CPSS practice class: Argument Parsing")

parser.add_argument('-s', '--str', type=str, default='hello world',
                    help='Prints some string.')

parser.add_argument('--multiple-args', type=int, nargs='+', default=[0],
                    help='Print multiple args.')

parser.add_argument('--must-required', action='store_true', required=True,
                    help='This argument is required.')

args = parser.parse_args()

print(f'String argument (-s/--str): {args.str}')
print(f'Multiple integers argument (--multiple-args): Type: {type(args.multiple_args)} | Args: {args.multiple_args}')
print(f'Required flag argument (--must-required): {args.must_required}')