# miniTRASGO analysis ancillary software
## C. Soneira, csoneira@ucm.es

### dat_treatment_caye/ contains software that goes inside mingo and is executed from the crontab:
- compress_and_clear: compresses all the files in the directory of minitrasgo cointaining the data in ascii (.dat) one by one.
- join.sh: asks for a data range (the same day is allowed) and collects all the data files, joins them and creates a new compressed file called merged_from_{start_date}_to_{end_date}.txt.tar.gz.
- logs_join.sh: takes all the logs and joins them in a file, then cleans it a little bit.

### Harvesting software:
- bring_and_extract.sh: takes the merged_from_{start_date}_to_{end_date}.txt.tar.gz compressed data file and brings it to the current computer to a directory called Data/, that is created if it does not exist, and extracts the datafile.
- logs_bring.sh: takes the joined log files with logs_join.sh from the minitrasgo to a folder, creating if non-existent, called Logs_and_Rates/

### Analysis software:
- main_analysis.py: the main file. Can be executed from the main directory. It creates the Results/ directory if it does not exist and also the Results/Results-from*/ directories automatically where all the outputs (text archives and plots) are stored. Can be executed from terminal when provided a Data/merged*.txt file or from the Spyder.
- loop_analysis.sh: to run the main_analysis.py through Data/
- plot_rate_and_temp.sh: takes the rates from the Results/Results-from-*/ to plot it with the temperature, taken from the Logs_and_Rates/ and then returns the plot in Logs_and_Rates/
- plot_logs.py: plots internal and external temperatures, relative humidity, pressure, voltage and gas flow taken from the Logs_and_Rates/ directory and puts the graphs on that same directory.

### Directories:
- Results/: created/stuffed automatically when executing main_analysis.py
- Data/: created/stuffed automatically when executing bring_and_extract.sh
- Logs_and_Rates/: created/stuffed automatically when executing logs_bring.sh
