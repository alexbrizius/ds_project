Intro
=====
Contained in the repository are the scripts and folder structure necessary to 
reproduce our presentation and term paper. The scripts contained have been tested
using python2.7.

Dependencies
===========
imblearn
matplotlib
pandas
pydot
sklearn

How to Run
==========
The easiest way to reproduce our work is to simply run:
    - './run_all.py LEAGUE_NAME' or './run_all_us.py LEAGUE_NAME'

Note: The 'us' affix on a script name indicitates that it uses undersampling. As 
discussed in our presentation and report, use undersampling for A and AA, whereas
do not use undersampling for AAA.

Results Location
===========
After './run_all.py' is run, look in 'images/' for our graphical results. Likewise, 
run './analyze.py' for numerical results in  a tabular format. Finally, look in
'LEAGUENAME_Results' for miscillanious intermediate files (like decision trees, etc.)
 
