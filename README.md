# README
Implemented the following tensorflow project: https://www.tensorflow.org/tutorials/quickstart/beginner
The gist of this small tutorial is that you construct a neural net with some amount of layers, where
at each step, the data is getting convolved on, in order to understand the image.  
As the image is analyzed by this NN, a probabilty distribution is eventually constructed,
by which we can vote on the state of some input.  

This code trains on the http://yann.lecun.com/exdb/mnist/ MNIST dataset, and essentially each 
index in our probability distribution represents which of the digits the input likely represents!

# TO RUN
Make sure you have Tensorflow installed.  If you don't, type:
```
pip3 install tensorflow
```

Then, simply clone this repository, and then run the program:
```
python3 ts_quickstart.py
```

If you do not have python installed, I'm no python tamer, so I'm not sure if this is the right place
for a python installation tutorial!  