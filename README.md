hyperbolic_wrapped_distribution
===

Code for reproducing results in ["A Wrapped Normal Distribution on Hyperbolic Space for Gradient-Based Learning"](https://arxiv.org/abs/1902.02992).

![Example of log density](/images/density_example.png)

## Setup Environment with Docker

```sh
$ docker build . -t <image-name>
$ docker run --runtime nvidia --rm -it \
  -v $HOME/data:/root/data \
  -v $PWD:/work -w /work \
  <image-name> bash
```

## Usage

train Hyperbolic VAE with synthetic dataset:

```sh
$ python3 -m scripts.train --recipe-path recipes/mlp_synthetic.yml \
  --p-z nagano <experiment-name>
```

train Hyperbolic VAE with MNIST dataset:

```sh
$ python3 -m scripts.train --p-z nagano <experiment-name>
```

train Hyperbolic VAE with Breakout dataset (you have to place [the dataset for explored trajectories of pretrained agent in Breakout](https://www.dropbox.com/s/hyq44euztzz23o8/breakout_states_v2.h5?dl=0) to `$HOME/data/breakout/state_samples`):

```sh
$ python3 -m scripts.train --recipe-path recipes/cnn_breakout.yml \
  --p-z nagano <experiment-name>
```

train Hyperbolic word embedding model with WordNet dataset:

```sh
$ python3 -m scripts.train_embedding --p-z nagano <experiment-name>
```
