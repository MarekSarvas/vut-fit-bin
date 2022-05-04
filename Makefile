
all:
	bash run.sh 2 2

plots:
	bash run.sh 3 3

dataset:
	bash run.sh 0 0 

setup:
	bash install.sh

reset_env:
	rm -r env-bin/
	bash install.sh
	
