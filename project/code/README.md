# Instructions on getting the project to run:

1. extract features for the mnist dataset. run:
	
	```bash
	cd /some/path/to/project/code/
	python3 feature_extraction.py -d mnist -e 12
    python3 feature_extraction.py -d cifar10 -e 100
	cd ../data/pickles/write
	find . -name "*mnist*" -exec mv {} ../read/ \;
    find . -name "*cifar10*" -exec mv {} ../read/ \;
	```

2. Generate quadruplets:
    ```bash
    # There should be quadruplets directory in both write and read directories
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
    
    **Copy+paste - mnist features:**
    _verify `quadruplets` directories existance!_
    ```bash
    cd /some/path/to/project/code/
    sh collect_mnist.sh
    sh collect_cifar10.sh
    ```
3. Run the pipeline
    _verify `results` directory existance and the path is updated in config.py_
    ```bash
    # There should be quadruplets directory in both write and read directories
    cd /some/path/to/project/code/
    python3 pipeline.py -d
    ```

    The plots are under the results directory
    The log file is under `log` directory
