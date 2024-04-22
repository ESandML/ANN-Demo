# Interactive Artificial Neural Network Function Approximation Demonstration

This notebook is provided as a learning resource to aid students of machine learning in understanding artificial neural networks (ANNs).

With this resource, you are able to visualise an animation of the approximation of a known target function, modelled by an ANN, as the network is trained on synthetic data representing the target function, as well as a decomposition of the final approximation. In this, you can experiment with different data and hyperparameters, such as the number of hidden layers, the number of hidden neurons, and the activation function, allowing for interactive learning. The visualisations are based on real Python code for training neural networks that is built in PyTorch, a popular library for deep learning, providing an accessible insight into the tools used to build today's advanced artificial intelligence systems.

## Animation of function approximated by an ANN throughout training

Fig. 1 shows example animations of the approximation of a known target function, modelled by an ANN, as the network is trained on synthetic data representing the target function, as allowed for by this resource. To demonstrate how this can be used for interactive and explorative learning, each animation shown is for a shallow neural network with 16 hidden neurons and the same initial weights, but different activation functions.

<p align="center">
  <img src="Images/relu.gif" width="300" />
  <img src="Images/elu.gif" width="300" /> <br />
  <img src="Images/leakyrelu.gif" width="300" />
  <img src="Images/gelu.gif" width="300" /> <br />
  <img src="Images/prelu.gif" width="300" />
  <img src="Images/celu.gif" width="300" /> <br />
  <img src="Images/relu6.gif" width="300" />
  <img src="Images/selu.gif" width="300" /> <br />
  <img src="Images/softshrink.gif" width="300" />
  <img src="Images/mish.gif" width="300" /> <br />
  <img src="Images/sigmoid.gif" width="300" />
  <img src="Images/softsign.gif" width="300" /> <br />
  <img src="Images/tanh.gif" width="300" />
  <img src="Images/tanhshrink.gif" width="300" /> <br />
	<em>
		Figure 1: Example animations of the approximation of a known target function, modelled by an ANN, as the network is trained on synthetic data representing the target function. Each animation is for a shallow neural network with 16 hidden neurons and the same initial weights, but different activation functions. Top to bottom, left to right: ReLU, LeakyReLU, PReLU, ReLU6, SoftShrink, Sigmoid, Tanh, ELU, GELU, CELU, SELU, Mish, SoftSign, Tanhshrink.
	</em>
</p>

## Visualisation of a detailed decomposition of approximation of target function by a shallow neural network

<p align="center">
  <img src="Images/shal_decomp.png" width="900" /> <br />
	<em>
		Figure 1: Example animations of the approximation of a known target function, modelled by an ANN, as the network is trained on synthetic data representing the target function. Each animation is for a shallow neural network with 16 hidden neurons and the same initial weights, but different activation functions. Top to bottom, left to right: ReLU, LeakyReLU, PReLU, ReLU6, SoftShrink, Sigmoid, Tanh, ELU, GELU, CELU, SELU, Mish, SoftSign, Tanhshrink.
	</em>
</p>
