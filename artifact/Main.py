# -*- coding: utf-8 -*-
"""
Main: This file provides instructions regarding how to run the experiments described in the main body of the paper.
"""


"""


.............................Figures.......................................................

To run the experiments:

Enter 5 for Fig. 5

Enter 6 for Fig. 6

Enter 7 for Fig. 7

Enter 8 for Fig. 8

Enter 9 for Fig. 9 
    
    
-------------------------------------Table-------------------------------------------------------------------------

To run the experiments:

Enter 10 for Tab. 1



--------------------------------------------------------------------------------------------
"""

# main.py
from DPMIX_Functions import DP_Mix
import sys

def main():
    # Check if an argument is provided
    if len(sys.argv) == 2:
        try:
            experiment_id = int(sys.argv[1])
        except ValueError:
            print("Please provide a valid integer for the experiment ID.")
            sys.exit(1)
    else:
        # Fallback to interactive input
        experiment_id = int(input("Please enter the ID of the experiment you wish to run: "))

    DP_Mix(experiment_id)

if __name__ == "__main__":
    main()




