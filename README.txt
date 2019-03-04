Installation of Python and Jython v2.7.0 is required for running these programs. 
	- Install Jython from https://www.jython.org/downloads.html
	- Set up path in environment variables (Windows)

1. Go to the directory, where the zip file has been extracted through cmd/terminal 
2. Run "dumper.py" in command line (python dumper.py)
	- This will create winequality_test, winequality_trg, winequality_val from winequality_white dataset
3. Run "NN_rhc.py", "NN_sa.py", "NN_ga.py", "tsp.py", "flipflop.py" and "continuouspeak.py" using "jython <filename>.py" 
	- Run one at a time or in different cmd/terminal windows
	- Last three files can take considerable amount of time to run, because of MIMIC algorithm implementation 
	- Logs will be made in the respective folders, such as NN_OUTPUT and TSP_LOGS
4. Run "plot.py" using " python plot.py"
5. All the graphs will be stored in "graphs" folder
