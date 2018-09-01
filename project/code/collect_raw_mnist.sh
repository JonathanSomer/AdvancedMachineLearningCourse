#!/usr/bin/env bash
python3 collect.py -d raw_mnist_0 -c 30;
python3 collect.py -d raw_mnist_1 -c 30;
python3 collect.py -d raw_mnist_2 -c 30;
python3 collect.py -d raw_mnist_3 -c 30;
python3 collect.py -d raw_mnist_4 -c 30;
python3 collect.py -d raw_mnist_5 -c 30;
python3 collect.py -d raw_mnist_6 -c 30;
python3 collect.py -d raw_mnist_7 -c 30;
python3 collect.py -d raw_mnist_8 -c 30;
python3 collect.py -d raw_mnist_9 -c 30;
mv ../data/pickles/write/raw_mnist* ../data/pickles/read/
mv ../data/pickles/write/quadruplets/raw_mnist* ../data/pickles/read/quadruplets/
