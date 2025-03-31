## Simple Convolutional Neural Network (CNN) in C++

This section presents a very basic implementation of a Convolutional Neural Network (CNN) in C++. It is designed to illustrate the fundamental concept of convolution and the use of an activation function (ReLU). Keep in mind that this is a highly simplified version and doesn't include many of the standard components you'd find in real-world CNNs.

### What is a Convolutional Neural Network (CNN)?

CNNs are a type of artificial neural network that are particularly good at processing data that has a grid-like structure, such as images. They're widely used for tasks like image recognition, object detection, and more. The core building block of a CNN is the convolutional layer.

### What is Convolution?

Imagine you have an image (represented as a grid of numbers) and a small "filter" or "kernel" (also a small grid of numbers). Convolution is like sliding this kernel over your input image, one position at a time. At each position, you perform an element-wise multiplication between the numbers in the kernel and the corresponding numbers in the image. Then, you sum up all these multiplied values to get a single number in the output. This output is called a "feature map."

Think of the kernel as a pattern detector. Different kernels can be designed to detect different features in an image, like edges, corners, or specific textures.

### Example: Convolution Operation

#### Input Image:
```
0 0 0 0 0
0 1 1 1 0
0 1 2 1 0
0 1 1 1 0
0 0 0 0 0
```

#### Convolutional Kernel:
```
1  0 -1
1  0 -1
1  0 -1
```

#### Output Feature Map (after convolution and ReLU):
```
0 0 0
3 3 0
6 3 0
3 3 0
0 0 0
```

### Explanation of the Output

- **Input Image:** This is our simple 5x5 grid. Notice the vertical line of `1`s and `2`s in the middle.
- **Convolutional Kernel:** As mentioned before, this kernel is designed to detect vertical edges.
- **Output Feature Map:** The resulting 3x3 grid shows where the kernel detected a vertical change in intensity in the input image. 
  - Higher positive values (like `3` and `6`) appear where there was a transition from `0` to `1` or `0` to `2` in the input image in the vertical direction.
  - The zeros indicate no strong vertical edge was detected at those locations (or the result was negative before ReLU). The ReLU activation then ensures all negative values become zero.

### Important Note

This is a very basic example with just one convolutional layer and ReLU. Real CNNs typically have multiple convolutional layers, pooling layers (to reduce the size of the feature maps), and fully connected layers at the end for making final predictions. Training a CNN involves feeding it lots of labeled data and adjusting the values in the kernels (the weights) so that the network learns to correctly identify patterns and make accurate predictions. This simple code just shows the forward pass for a single, pre-defined kernel.

---

## Simple Recurrent Neural Network (RNN) in C++

This section provides a basic implementation of a Recurrent Neural Network (RNN) in C++. It is designed to introduce the fundamental concept of recurrent connections and how information flows through an RNN.

### What is a Recurrent Neural Network (RNN)?

RNNs are a class of artificial neural networks that excel at processing sequential data, such as time series, natural language, and speech recognition. Unlike traditional neural networks, RNNs maintain a hidden state that allows them to retain information from previous time steps, making them ideal for sequence-based tasks.

### How Does an RNN Work?

An RNN processes input data one time step at a time, updating its hidden state based on the current input and the previous hidden state. Mathematically, the RNN updates its hidden state as follows:

\[ h_t = \tanh(W_h h_{t-1} + W_x x_t + b) \]

where:
- \( h_t \) is the hidden state at time step \( t \),
- \( x_t \) is the input at time step \( t \),
- \( W_h \) and \( W_x \) are weight matrices,
- \( b \) is the bias,
- \( \tanh \) is the activation function.

### Example: Simple RNN Computation

#### Given Inputs:
```
Sequence: [1.0, 2.0, 3.0]
Initial Hidden State: 0.0
```

#### Weight and Bias:
```
W_h = 0.5
W_x = 0.8
b = 0.1
```

#### Hidden State Computation:
```
h_1 = tanh(0.5 * 0.0 + 0.8 * 1.0 + 0.1) = tanh(0.9) ≈ 0.716
h_2 = tanh(0.5 * 0.716 + 0.8 * 2.0 + 0.1) = tanh(1.958) ≈ 0.888
h_3 = tanh(0.5 * 0.888 + 0.8 * 3.0 + 0.1) = tanh(2.644) ≈ 0.989
```

### Explanation of the Output

- The hidden state evolves over time based on past values and new inputs.
- The tanh function ensures the values remain between -1 and 1.
- The hidden state carries memory from previous inputs, making RNNs useful for sequential data processing.

### Example Output of the Program

```
Hidden States:
Time step 0: [-0.554604 0.551775 ]
Time step 1: [0.00848008 -0.601971 ]
Time step 2: [0.762038 -0.611544 ]
Time step 3: [0.907159 -0.793023 ]
```

### Important Note

This is a simple example with a single-layer RNN. Real-world RNNs often include multiple layers, more complex architectures like LSTMs (Long Short-Term Memory) or GRUs (Gated Recurrent Units), and trained weight matrices instead of fixed values.

This basic code provides insight into how an RNN processes sequential data step by step.