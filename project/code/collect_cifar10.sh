#!/usr/bin/env bash
python3 collect.py -d cifar10_0 -c 30;
python3 collect.py -d cifar10_1 -c 30;
python3 collect.py -d cifar10_2 -c 30;
python3 collect.py -d cifar10_3 -c 30;
python3 collect.py -d cifar10_4 -c 30;
python3 collect.py -d cifar10_5 -c 30;
python3 collect.py -d cifar10_6 -c 30;
python3 collect.py -d cifar10_7 -c 30;
python3 collect.py -d cifar10_8 -c 30;
python3 collect.py -d cifar10_9 -c 30;
python3 collect.py -d cifar10 -c 30;
mv ../data/pickles/write/cifar10* ../data/pickles/read/
mv ../data/pickles/write/quadruplets/cifar10* ../data/pickles/read/quadruplets/
