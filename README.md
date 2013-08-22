KGMindep_SympyTheanoOp
======================

Kinetic Global Modeling (KGM) framework independent testing of adding a custom theano Op (wrapping a scipy spline routine) to be mapped by the sympy.printing.theanocode function

Run Sympy_Theano_KGM_indep.py for test cases (will try testcases when run as the main file). First two test cases are known to work (and moreso for my learning), third case (f2) is my problem.

loc_theanocode.py is a slightly modified (including my verbose statements to try and figure out what is happening) version of sympy.printing.theanocode
