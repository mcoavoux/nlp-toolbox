

"""
Adds additional uppercase entries to dictionaries
"""
import sys

for line in sys.stdin:
    sys.stdout.write(line)
    sys.stdout.write(line[0].upper()+line[1:])
    
