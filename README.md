# recursive-sgd
A proof of concept of a recursion doing stochastic gradient descent done in Python

## Why?
IDK I just hit me one day that, if I represent a neural network as a sequence of transformations, I could train it recursively.
## How?
The idea is simple: to train some layer of a neural network, a recursive function trains the next layer and returns the gradient of that next layer. This gradient is then used to update the current layer and calculate the gradient w.r.t. the inputs of the current layer, which is then returned to a previous layer.
The gist of it is in the `sgd_step` function of `recursive_sgd/sgd.py`.

## Once again, why?
*The answer is left as an exercise to the reader*

## Installation
Ordinary `pip3 install recursive-sgd` does the trick.
Alternatively, one can use:
```
git clone https://github.com/InCogNiTo124/recursive-sgd.git
cd recursive-sgd
python3 setup.py install
```
## Usage
There's a CLI available.
### Training
`python3 cli.py train [OPTIONS]` where `OPTIONS` can be the following:
- `-d FILEPATH` or `--dataset FILEPATH` - CSV dataset
- `-i INT` or `--input-size INT` - the number of input features
- `--lr FLOAT` - learning rate
- `--loss VALUE` - Loss function (`CE` for CrossEntropy, `MSE` for MeanSquaredError)
- `-e INT` or `--epochs INT` - the number of epochs
- `--batch-size INT` - the size of one batch
- `--shuffle` - shuffle dataset after every epoch (default)
- `--no-shuffle` - never shuffle

The architecture is defined with arguments as well:
- `-l SIZE` - a new layer with `SIZE` neurons
- `-b` - add bias
- `-s` - add Sigmoid activation
- `-r` - add ReLU activation
- `-t` - add Tanh activation



Checkout the example at `train_command`
### Testing
`python3 cli.py test [OPTIONS]` where `OPTIONS` can be the following:
- `-m FILEPATH` or `--model FILEPATH` - the path to the saved model
- `-d FILEPATH` or `--dataset FILEPATH` - CSV dataset
- `-i INT` or `--input-size INT` - the number of input features
- `--metrics VALUE` - metric with witch you wish to test the model with.

Checkout the example at `test_command`

## Notes
- After training, the model will be saved in `$PWD` as `MODEL.sgd`. This is hardcoded for now, but will be configurable in the future.
- There is no `-h` nor `--help` flag. I am parsing the arguments myself without any framework at all and I didn't bother writing help *in the CLI* but here.
- There are serious limitations in the dataset loading:
	- Only CSV format is allowed
	- The columns **MUST** be separated by `,` character only.
	- The true labels column is implicitly the last one
		- Technically it's every remaining column after the number of features defined by `-i` or `--input-size` which may introduce subtle bugs of having more than 1 target variables.
- Only available metric at the moment is `accuracy`.
### One final note
https://twitter.com/johnwilander/status/1176457013305303040

