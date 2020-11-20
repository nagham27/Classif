Description:
The ooclassifier.py script reads in a training dataset from a file,and classifies it
using hardcoded target words from the user or using certain positive features in the file,
or using a number(specified by the user) of the most frequently occuring words in positive 
taining instances of a training dataset.

The sys module was imported,since the use of sys.stdin and sys.argv was required.
The copy module was imported,since the use of copy.deepcopy() was required.


Running instructions:

if used without a driver file:
run python3 ooclassifier.py <file_name> in the terminal.

if used with a driver file:
run python3 <driver_file> <file_name> in the terminal.

Note: the driver script should import ooclassifier using "from ooclassifier import * ".


Assumptions:

1)Training instances in the file have labels that start with '#'.

2)Comments start with '%' in the file.

3)pos-features in a file start with '%pos-features' followed by all positive features.

4)pos-label in a file starts with '%pos-label' followed by the positive label.
 
