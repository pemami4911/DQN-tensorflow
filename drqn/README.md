# Deep Recurrent Q-Network

Tensorflow implementation of [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/pdf/1507.06527v3.pdf).

You can test both DQN and DRQN on OpenAI gym's Atari environments, as well as OpenAI gym's Doom environments. I am using [ppaquette/gym-doom](https://github.com/ppaquette/gym-doom) as the interface with VizDoom. 

You can change the sequence length that DRQN uses as input in `config.py`. By default it is set to 10. Learning occurs only for the last 6 states in the sequence; the first 4 are used to "prime" the hidden state of the LSTM. You'll need to modify `init_sequence_length`, the initialization length for the sequence, if you also modify the seq length. 

The `history_length` is fixed at 1 for DRQN; all input states are grayscale images- this implementation doesn't support RGB for DRQN. In other words, the input sequence is a tensor of the last 10 grayscale images of the states, and the `history_length` dimension of 1 is fed in to the convolutional layers as the number of channels. 

To run an experiment with DRQN on Atari Breakout-v0 without GPU and with display: 

    python main.py --model=drqn --env_name=Breakout-v0 --env_type=simple --use_gpu=False --display=True

To run an experiment with DRQN on the Doom env DefendCenter-v0 with GPU and no display: 
    
    python main.py --model=drqn --env_name=ppaquette/DoomDefendCenter-v0 --env_type=Doom --use_gpu=True --gpu_fraction=2/3

You can run experiments with DQN by changing the model to `dqn`. 

You can view the Tensorflow computation graph with Tensorboard for DQN and DRQN, which is helpful for debugging. 

## Requirements

This branch doesn't use OpenCV2 (installing it if you don't have it just for resizing an image and converting it to grayscale is overkill), and instead I use skimage.
`gym-pull` is only needed if you're going to use the Doom environments - I had some trouble installing this on OSX, but it seems to work fine on Ubuntu 14.04.  

- Python 2.7 or Python 3.3+
- [gym](https://github.com/openai/gym)
- [gym-pull](https://github.com/ppaquette/gym-pull) 
- [tqdm](https://github.com/tqdm/tqdm)
- [skimage](http://scikit-image.org/)
- [TensorFlow](https://www.tensorflow.org/)
