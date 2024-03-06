#!/bin/bash
repo_dir=$(realpath $(dirname .))
proj_name=$(basename $repo_dir)
pdflatex -output-directory=${repo_dir}/report ${repo_dir}/report/${proj_name}.tex
bibtex ${repo_dir}/report/${proj_name}.aux
pdflatex -output-directory=${repo_dir}/report ${repo_dir}/report/${proj_name}.tex
pdflatex -output-directory=${repo_dir}/report ${repo_dir}/report/${proj_name}.tex