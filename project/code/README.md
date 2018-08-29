# Instructions on getting the project to run:

1. extract features for the mnist dataset. run:
	
	```
	cd /some/path/to/project/code/
	python3 feature_extraction.py -d mnist
	cd ../data/pickles/write
	find . -name "*mnist*" -exec mv {} ../read/ \;
	```

2. Generate quadruplets: TODO URI
