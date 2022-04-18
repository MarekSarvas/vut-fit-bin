
all:
	bash run.sh

dataset:
	bash run.sh 0 0 

setup:
	bash install.sh

reset_env:
	rm -r env-bin/
	bash install.sh
	
