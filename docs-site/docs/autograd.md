---
id: autograd
title: Autograd in SINGA
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

There are two typical ways to implement autograd, via symbolic differentiation like [Theano](http://deeplearning.net/software/theano/index.html) or reverse differentiation like [Pytorch](https://pytorch.org/docs/stable/notes/autograd.html). Singa follows Pytorch way, which records the computation graph and apply the backward propagation automatically after forward propagation. The autograd algorithm is explained in details [here](https://pytorch.org/docs/stable/notes/autograd.html). We explain the relevant modules in Singa and give an example to illustrate the usage.

## Relevant Modules

There are three classes involved in autograd, namely `singa.tensor.Tensor`, `singa.autograd.Operation`, and `singa.autograd.Layer`. In the rest of this article, we use tensor, operation and layer to refer to an instance of the respective class.

### Tensor

Three attributes of Tensor are used by autograd,

- `.creator` is an `Operation` instance. It records the operation that generates the Tensor instance.
- `.requires_grad` is a boolean variable. It is used to indicate that the autograd algorithm needs to compute the gradient of the tensor (i.e., the owner). For example, during backpropagation, the gradients of the tensors for the weight matrix of a linear layer and the feature maps of a convolution layer (not the bottom layer) should be computed.
- `.stores_grad` is a boolean variable. It is used to indicate that the gradient of the owner tensor should be stored and output by the backward function. For example, the gradient of the feature maps is computed during backpropagation, but is not included in the output of the backward function.

Programmers can change `requires_grad` and `stores_grad` of a Tensor instance. For example, if later is set to True, the corresponding gradient is included in the output of the backward function. It should be noted that if `stores_grad` is True, then `requires_grad` must be true, not vice versa.

### Operation

It takes one or more `Tensor` instances as input, and then outputs one or more `Tensor` instances. For example, ReLU can be implemented as a specific Operation subclass. When an `Operation` instance is called (after instantiation), the following two steps are executed:

1. record the source operations, i.e., the `creator`s of the input tensors.
2. do calculation by calling member function `.forward()`

There are two member functions for forwarding and backwarding, i.e., `.forward()` and `.backward()`. They take `Tensor.data` as inputs (the type is `CTensor`), and output `Ctensor`s. To add a specific operation, subclass `operation` should implement their own `.forward()` and `.backward()`. The `backward()` function is called by the `backward()` function of autograd automatically during backward propogation to compute the gradients of inputs (according to the `require_grad` field).

### Layer

For those operations that require parameters, we package them into a new class, `Layer`. For example, convolution operation is wrapped into a convolution layer. `Layer` manages (stores) the parameters and calls the corresponding `Operation`s to implement the transformation.

## Examples

Multiple examples are provided in the [example folder](https://github.com/apache/singa/tree/master/examples/autograd). We explain two representative examples here.

### Operation only

The following codes implement a MLP model using only Operation instances (no Layer instances).

#### Import packages

```python
from singa.tensor import Tensor
from singa import autograd
from singa import opt
```

#### Create weight matrix and bias vector

The parameter tensors are created with both `requires_grad` and `stores_grad` set to `True`.

```python
w0 = Tensor(shape=(2, 3), requires_grad=True, stores_grad=True)
w0.gaussian(0.0, 0.1)
b0 = Tensor(shape=(1, 3), requires_grad=True, stores_grad=True)
b0.set_value(0.0)

w1 = Tensor(shape=(3, 2), requires_grad=True, stores_grad=True)
w1.gaussian(0.0, 0.1)
b1 = Tensor(shape=(1, 2), requires_grad=True, stores_grad=True)
b1.set_value(0.0)
```

#### Training

```python
inputs = Tensor(data=data)  # data matrix
target = Tensor(data=label) # label vector
autograd.training = True    # for training
sgd = opt.SGD(0.05)   # optimizer

for i in range(10):
    x = autograd.matmul(inputs, w0) # matrix multiplication
    x = autograd.add_bias(x, b0)    # add the bias vector
    x = autograd.relu(x)            # ReLU activation operation

    x = autograd.matmul(x, w1)
    x = autograd.add_bias(x, b1)

    loss = autograd.softmax_cross_entropy(x, target)

    for p, g in autograd.backward(loss):
        sgd.update(p, g)
```

### Operation + Layer

The following [example](https://github.com/apache/singa/blob/master/examples/autograd/mnist_cnn.py) implements a CNN model using layers provided by the autograd module.

#### Create the layers

```python
conv1 = autograd.Conv2d(1, 32, 3, padding=1, bias=False)
bn1 = autograd.BatchNorm2d(32)
pooling1 = autograd.MaxPool2d(3, 1, padding=1)
conv21 = autograd.Conv2d(32, 16, 3, padding=1)
conv22 = autograd.Conv2d(32, 16, 3, padding=1)
bn2 = autograd.BatchNorm2d(32)
linear = autograd.Linear(32 * 28 * 28, 10)
pooling2 = autograd.AvgPool2d(3, 1, padding=1)
```

#### Define the forward function

The operations in the forward pass will be recorded automatically for backward propagation.

```python
def forward(x, t):
    # x is the input data (a batch of images)
    # t the the label vector (a batch of integers)
    y = conv1(x)           # Conv layer
    y = autograd.relu(y)   # ReLU operation
    y = bn1(y)             # BN layer
    y = pooling1(y)        # Pooling Layer

    # two parallel convolution layers
    y1 = conv21(y)
    y2 = conv22(y)
    y = autograd.cat((y1, y2), 1)  # cat operation
    y = autograd.relu(y)           # ReLU operation
    y = bn2(y)
    y = pooling2(y)

    y = autograd.flatten(y)        # flatten operation
    y = linear(y)                  # Linear layer
    loss = autograd.softmax_cross_entropy(y, t)  # operation
    return loss, y
```

#### Training

```python
autograd.training = True
for epoch in range(epochs):
    for i in range(batch_number):
        inputs = tensor.Tensor(device=dev, data=x_train[
                               i * batch_sz:(1 + i) * batch_sz], stores_grad=False)
        targets = tensor.Tensor(device=dev, data=y_train[
                                i * batch_sz:(1 + i) * batch_sz], requires_grad=False, stores_grad=False)

        loss, y = forward(inputs, targets) # forward the net

        for p, gp in autograd.backward(loss):  # auto backward
            sgd.update(p, gp)
```

### Supported operators

#### mean

Init a element-wise Mean operator.

##### Input

- l: a list of CTensor

##### Output

- the output CTensor.

#### relu

Init a ReLU operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### less

Init a element-wise Less (x&lt;derp>y) operator.

##### Input

- x: CTensor, input tensor.
- y: CTensor, input tensor.

##### Output

- the output CTensor.

#### clip

Init a Clip operator.

##### Input

- x: CTensor, input tensor.
- min: float, the min value. If not set, the operator will not clip the min value.
- max: float, the max value. If not set, the operator will not clip the max value.

##### Output

- the output CTensor.

#### identity

Init a Identity operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### matmul

Init a matrix multiplication operator.

##### Input

- x: CTensor, input tensor.
- w: CTensor, input tensor.

##### Output

- the output CTensor.

#### greater

Init a element-wise Greater (x>y) operator.

##### Input

- x: CTensor, input tensor.
- y: CTensor, input tensor.

##### Output

- the output CTensor.

#### add_bias

Add bias to each row / column of the Tensor, depending on the axis arg.

##### Input

- x: CTensor, input tensor.
- b: CTensor, input tensor.
- axis: int, 0 or 1, default is 0.

##### Output

- the output CTensor.

#### reshape

Init a Reshape operator.

##### Input

- x: CTensor, input tensor.
- shape: list of int, the new shape.

##### Output

- the output CTensor.

#### prelu

Init a PRelu operator. This operator supports [Multidirectional Broadcasting](https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md).

##### Input

- x: CTensor, input tensor.
- slope: CTensor, slope tensor.

##### Output

- the output CTensor.

#### add

Init a Add operator. This operator supports [Multidirectional Broadcasting](https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md).

##### Input

- x: CTensor, input tensor.
- b: CTensor, input tensor.

##### Output

- the output CTensor.

#### elu

Init a Elu operator.

##### Input

- x: CTensor, input tensor.
- alpha: float, alpha value.

##### Output

- the output CTensor.

#### Equal

Init a element-wise Equal (x=y) operator.

##### Input

- x: CTensor, input tensor.
- y: CTensor, input tensor.

##### Output

- the output CTensor.

#### selu

Init a SeLU operator.

##### Input

- x: CTensor, input tensor.
- alpha: float, alpha value.
- gamma: float, alpha value.

##### Output

- the output CTensor.

#### softmax

Init a SoftMax operator.

##### Input

- x: CTensor, input tensor.
- axis: int, the axis along with.

##### Output

- the output CTensor.

#### sum

Init a element-wise Sum operator.

##### Input

- l: a list of CTensor

##### Output

- the output CTensor.

#### cross_entropy

Init a CrossEntropy operator.

##### Input

- y: CTensor, 1d or 2d tensor, the prediction data(output) of current network.
- t: CTensor, 1d or 2d tensor, the target data for training.

##### Output

- the output CTensor.

#### softmax_cross_entropy

Init a SoftMaxCrossEntropy operator.

##### Input

- y: CTensor, 1d or 2d tensor, the prediction data(output) of current network.
- t: CTensor, 1d or 2d tensor, the target data for training.

##### Output

- the output CTensor.

#### MeanSquareError

Init a MeanSquareError operator.

##### Input

- y: CTensor, 1d or 2d tensor, the prediction data(output) of current network.
- t: CTensor, 1d or 2d tensor, the target data for training.

##### Output

- the output CTensor.

#### flatten

Init a Flatten operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### cat

Init a Concat operator.

##### Input

- x: CTensor, input tensor.
- axis: int, the axis along with.

##### Output

- the output CTensor.

#### conv2d

Init a \_Conv2d operator.

##### Input

- handle: ConvHandle for cpu or CudnnConvHandle for gpu
- x: CTensor, input tensor.
- W: CTensor, weight
- b: CTensor, bias
- odd_padding: tuple of four bins, the odd paddding is the value that cannot be handled by the tuple padding (w, h) mode, so we need to firstly handle the input, then use the nomal padding method.

##### Output

- the output CTensor.

#### batchnorm_2d

Init a \_BatchNorm2d operator.

##### Input

- handle,
- x, CTensor, input tensor.
- scale, CTensor, scale.
- bias, CTensor, bias.
- running_mean, float, running mean.
- running_var, float running variance.

##### Output

- the output CTensor.

#### Pooling2d

Init a \_Pooling2d operator.

##### Input

- handle: ConvHandle for cpu or CudnnConvHandle for gpu
- x: CTensor, input tensor.
- odd_padding: tuple of four bins, the odd paddding is the value that cannot be handled by the tuple padding (w, h) mode, so we need to firstly handle the input, then use the nomal padding method.

##### Output

- the output CTensor.

#### tanh

Init a Tanh operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### cos

Init a Cos operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### cosh

Init a Cosh operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### acos

Init a Acos operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### acosh

Init a Acosh operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### sin

Init a Sin operator.

###### Input

- x: CTensor, input tensor.

###### Output

- the output CTensor.

#### sinh

Init a Sinh operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### asin

Init a Asin operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### asinh

Init a Asinh operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### tan

Init a Tan operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### atan

Init a Atan operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### atanh

Init a Atanh operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### sigmoid

Init a Sigmoid operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### Mul

Init a do pointwise multiplication operator. This operator supports [Multidirectional Broadcasting](https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md).

##### Input

- x: CTensor, input tensor.
- y: CTensor, input tensor.

##### Output

- the output CTensor.

#### unsqueeze

Init a Unsqueeze operator.

##### Input

- x: CTensor, input tensor.
- axis: int, the axis along with. The default is -1.

##### Output

- the output CTensor.

#### transpose

Init a Transpose operator.

##### Input

- x: CTensor, input tensor.
- shape: list of int, the new shape.

##### Output

- the output CTensor.

#### abs

Init a Abs operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### exp

Init a Exp operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### leakyrelu

Init a LeakyRelu operator.

##### Input

- x: CTensor, input tensor.
- a: float, alpha value.

##### Output

- the output CTensor.

#### sign

Init a Sign operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### pow

Init a element-wise Pow operator. This operator supports [Multidirectional Broadcasting](https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md).

##### Input

- x: CTensor, input tensor.
- y: CTensor, input tensor.

##### Output

- the output CTensor.

#### SoftSign

Init a SoftSign operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### sqrt

Init a Sqrt operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### SoftPlus

Init a SoftPlus operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### sub

Init a element-wise Sub operator. This operator supports [Multidirectional Broadcasting](https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md).

##### Input

- x: CTensor, input tensor.
- y: CTensor, input tensor.

##### Output

- the output CTensor.

#### min

Init a element-wise Min operator.

##### Input

- l: a list of CTensor

##### Output

- the output CTensor.

#### log

Init a Log operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### hardsigmoid

Init a HardSigmoid operator.

##### Input

- x: CTensor, input tensor.
- alpha: float, alpha value.
- gamma: float, alpha value.

##### Output

- the output CTensor.

#### squeeze

Init a Squeeze operator.

##### Input

- x: CTensor, input tensor.
- axis: list of int, the axis along with.

##### Output

- the output CTensor.

#### div

Init a element-wise Div operator. This operator supports [Multidirectional Broadcasting](https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md).

##### Input

- x: CTensor, input tensor.
- y: CTensor, input tensor.

##### Output

- the output CTensor.

#### shape

Init a Shape operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### max

Init a element-wise Max operator.

##### Input

- l: a list of CTensor

##### Output

- the output CTensor.

#### \_and

Init a element-wise And operator.

##### Input

- x: CTensor, input tensor.
- y: CTensor, input tensor.

##### Output

- the output CTensor.

#### \_or

Init a element-wise Or operator.

##### Input

- x: CTensor, input tensor.
- y: CTensor, input tensor.

##### Output

- the output CTensor.

#### \_not

Init a element-wise Not operator.

##### Input

- x: CTensor, input tensor.
- y: CTensor, input tensor.

##### Output

- the output CTensor.

#### \_xor

Init a element-wise Xor operator.

##### Input

- x: CTensor, input tensor.
- y: CTensor, input tensor.

##### Output

- the output CTensor.

#### negative

Init a Negative operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### reciprocal

Init a Reciprocal operator.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### gemm

Init a General Matrix multiplication(Gemm) operator, Compute Y = alpha _ A' _ B' + beta \* C, where input tensor A has shape (M, K) or (K, M), input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N), and output tensor Y has shape (M, N).

- A' = transpose(A) if transA else A
- B' = transpose(B) if transB else B

This operator supports [Multidirectional Broadcasting](https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md).

##### Input

- A: tensor, The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.
- B: tensor, The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
- C: tensor(optional), Optional input tensor C. If not specified, the computation is done as if C is a scalar 0. The shape of C should be unidirectional broadcastable to (M, N).
- alpha: float, Scalar multiplier for the product of input tensors A \* B.
- beta: float, Scalar multiplier for input tensor C.
- transA: int, Whether A should be transposed
- transB: int, Whether B should be transposed

##### Output

- tensor, the output

#### constantOfShape

Init a ConstantOfShape, generate a tensor with given value and shape.

##### Input

- x: CTensor, 1D tensor. The shape of the expected output tensor. All values must be >= 0.
- value: (Optional) The value of the output elements. Should be a one-element tensor. If not specified, it defaults to a tensor of value 0 and datatype float32

##### Output

- the output CTensor. If attribute 'value' is specified, the value and datatype of the output tensor is taken from 'value'. If attribute 'value' is not specified, the value in the output defaults to 0, and the datatype defaults to float32.

#### dropout

Init a Dropout, which scales the masked input data by the following equation: output = scale _ data _ mask, scale = 1. / (1. - ratio).

##### Input

- x: CTensor, input tensor.
- ratio: float, he ratio of random dropout, with value in [0, 1).

##### Output

- the output CTensor.

#### reduceSum

Init a ReduceSum, computes the sum of the input tensor's element along the provided axes.

##### Input

- x: CTensor, input tensor.
- axes: list of ints, A list of integers, along which to reduce. Accepted range is [-r, r-1] where r = rank(data). The default is None, which reduces over all the dimensions of the input tensor.
- keepdims: int, Keep the reduced dimension or not, default 1 mean keep reduced dimension.

##### Output

- the output CTensor.

#### reduceMean

Init a ReduceMean, computes the mean of the input tensor's element along the provided axes.

##### Input

- x: CTensor, input tensor.
- axes: list of ints, A list of integers, along which to reduce. Accepted range is [-r, r-1] where r = rank(data). The default is None, which reduces over all the dimensions of the input tensor.
- keepdims: int, Keep the reduced dimension or not, default 1 mean keep reduced dimension.

##### Output

- the output CTensor.

#### slice

Init a Slice, Produces a slice of the input tensor along multiple axes. Similar to numpy: https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html

##### Input

- x: CTensor, input tensor.
- starts: list of ints, starting indices of corresponding axis
- ends: list of ints, ending indices of corresponding axis
- axes: list of ints, axes that `starts` and `ends` apply to. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).
- steps: list of ints, slice step of corresponding axis in `axes`. Negative value means slicing backward. 'steps' cannot be 0. Defaults to 1.

##### Output

- the output CTensor.

#### ceil

Ceil takes one input data (Tensor) and produces one output data (Tensor) where the ceil is, y = ceil(x), is applied to the tensor elementwise.

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### split

Init a Split, Split a tensor into a list of tensors, along the specified 'axis'.

##### Input

- x: CTensor, input tensor.
- axis: int, Which axis to split on. A negative value means counting dimensions from the back. Accepted range is [-rank, rank-1] where r = rank(input).
- parts: list of ints, length of each output, which can be specified using argument 'parts'. Otherwise, the tensor is split to equal sized parts.

##### Output

- the output CTensor.

#### gather

Init a Gather, Given data tensor of rank r >= 1, and indices tensor of rank q, gather entries of the axis dimension of data (by default outer-most one as axis=0) indexed by indices, and concatenates them in an output tensor of rank q + (r - 1).

##### Input

- x: CTensor, input tensor.
- axis: int, Which axis to slice on. A negative value means counting dimensions from the back. Accepted range is [-rank, rank-1] where r = rank(input).
- indices: list of ints, entries of the axis dimension of data.

##### Output

- the output CTensor.

#### tile

Init a Tile, Constructs a tensor by tiling a given tensor. This is the same as function tile in Numpy: https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html

##### Input

- x: CTensor, input tensor.
- repeats: 1D int64 matrix of the same length as input's dimension number, includes numbers of repeated copies along input's dimensions.

##### Output

- the output CTensor.

#### nonzero

Returns the indices of the elements that are non-zero (in row-major order - by dimension). NonZero behaves similar to numpy.nonzero: https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html

##### Input

- x: CTensor, input tensor.

##### Output

- the output CTensor.

#### cast

The operator casts the elements of a given input tensor to a data type specified by the 'to' argument and returns an output tensor of the same size in the converted type.

##### Input

- x: CTensor, input tensor.
- to: data type

##### onehot

Produces a one-hot tensor based on inputs.

#### onehot

The operator casts the elements of a given input tensor to a data type specified by the 'to' argument and returns an output tensor of the same size in the converted type.

##### Input

- axis: Axis along which one-hot representation in added. Default: axis=-1. axis=-1 means that the additional dimension will be inserted as the innermost/last dimension in the output tensor.
- indices: Scalar specifying the number of classes in one-hot tensor. This is also the size of the one-hot dimension (specified by 'axis' attribute) added on in the output tensor. The values in the 'indices' input tensor are expected to be in the range [-depth, depth-1]. In case 'depth' is of non-integer type, it will be casted to int64 before use.
- values: Rank 1 tensor containing exactly two elements, in the format [off_value, on_value], where 'on_value' is the value used for filling locations specified in 'indices' input tensor, and 'off_value' is the value used for filling locations other than those specified in 'indices' input tensor.

##### Output

- the output CTensor.
