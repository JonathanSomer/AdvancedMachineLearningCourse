# Instructions on getting the project to run:

1. extract features for the mnist dataset. run:
	
	```bash
	cd /some/path/to/project/code/
	python3 feature_extraction.py -d mnist
	cd ../data/pickles/write
	find . -name "*mnist*" -exec mv {} ../read/ \;
	```

2. Generate quadruplets:
    ```bash
    cd /some/path/to/project/code/
    python3 collect.py -d {dataset name} -c {number of clusters}
    cd ../data/pickles/write
    mv mnist* ../read/
    mv quadruplets/mnist* ../read/quadruplets/
    ```
    For more options: `python3 collect.py --help`
    
    In order to generate quadruplets without some category `k`, `_k` 
    should be appended to the dataset name i.e. `mnist_3`.
    
    In order to generate quadruplets for the raw data for a dataset, 
    `raw_` should be appended to the left of dataset name i.e. `raw_mnist`.
    
    Those also can be combined i.e. `raw_mnist_3`.