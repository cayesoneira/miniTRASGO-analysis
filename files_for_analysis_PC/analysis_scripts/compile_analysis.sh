#!/bin/bash

directory_to_check="cython_optimized_analysis"

if [ ! -d "$directory_to_check" ]; then
	# Directory exists, perform some action
	echo "Directory exists: $directory_to_check"

	mkdir cython_optimized_analysis

	# Create a cython copy of the main file
	cp main_analysis.py ./cython_optimized_analysis/main_analysis_cy.pyx

	cd cython_optimized_analysis

	# Define the text content for setup.py
	setup_py_content=$(cat <<EOL
	from setuptools import setup
	from Cython.Build import cythonize

	setup(
	    ext_modules = cythonize("main_analysis_cy.pyx")
	)
	EOL
	)

	# Write the content to setup.py
	echo "$setup_py_content" > setup.py

	echo "setup.py has been created."

	# Compile the python script
	python3 setup.py build_ext --inplace

	rm setup.py
fi

# Define the text content for execute_optimized_analysis.py
execute_optimized_analysis_content=$(cat <<EOL
import ./cython_optimized_analysis/main_analysis_cy
EOL
)

# Write the content to execute_optimized_analysis.py
echo "$execute_optimized_analysis_content" > ./execute_optimized_analysis.py

python3 execute_optimized_analysis.py

rm execute_optimized_analysis.py
