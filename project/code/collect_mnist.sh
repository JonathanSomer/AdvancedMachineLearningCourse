python3 collect.py -d mnist_0 -c 30;
python3 collect.py -d mnist_1 -c 30;
python3 collect.py -d mnist_2 -c 30;
python3 collect.py -d mnist_3 -c 30;
python3 collect.py -d mnist_4 -c 30;
python3 collect.py -d mnist_5 -c 30;
python3 collect.py -d mnist_6 -c 30;
python3 collect.py -d mnist_7 -c 30;
python3 collect.py -d mnist_8 -c 30;
python3 collect.py -d mnist_9 -c 30;
mv ../data/pickles/write/mnist* ../data/pickles/read/
mv ../data/pickles/write/quadruplets/mnist* ../data/pickles/read/quadruplets/
