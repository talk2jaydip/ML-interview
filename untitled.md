

### 📚 MNIST Dataset Overview

- **Image Size**:28 × 28 pixel
- **Channels**:1 (grayscale
- **Classes**:10 (digits 0 through 9

---

### 🧠 Convolutional Layer Parameters

Let's define a convolutional layer with the following parameters:

- **Input Channels** 1
- **Output Channels (Filters)** 
- **Kernel Size** 3 ×3
- **Stride** 1
- **Padding** 0 (no paddin)

---

### 🧮 Calculating Output Dimensions

The output dimensions of a convolutional layer can be calculated using the formula

$$
\text{Output Size} = \left\lfloor \frac{\text{Input Size} - \text{Kernel Size} + 2 \times \text{Padding}}{\text{Stride}} \right\rfloor + 1$$

Applying this to our example:

- **Input Size*: 28
- **Kernel Size*:3
- **Padding*:0
- **Stride*:1

$$\text{Output Size} = \left\lfloor \frac{28 - 3 + 0}{1} \right\rfloor + 1 = 26$$

So, the output feature map will have dimensios 26 ×26, and with 32 filters, the output shape becoms (32, 26, 6).

---

### 🧪 PyTorch Implementation

Here's how you can implement this in PyTorc:


```python
import torch
import torch.nn as nn

# Define the convolutional layer
conv_layer = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0)

# Create a dummy input tensor with shape (batch_size=1, channels=1, height=28, width=28)
input_tensor = torch.randn(1, 1, 28, 28)

# Apply the convolutional layer
output_tensor = conv_layer(input_tensor)

# Print the output shape
print("Output shape:", output_tensor.shape)
``


**Expected Output*:


```
Output shape: torch.Size([1, 32, 26, 26])
``


This confirms our manual calculation.

---

### 🔄 Adding a MaxPooling Layer

To further reduce the spatial dimensions, we can add a MaxPooling layr:


```python
# Define a MaxPooling layer with kernel size 2 and stride 2
pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

# Apply the pooling layer
pooled_output = pool_layer(output_tensor)

# Print the pooled output shape
print("Pooled output shape:", pooled_output.shape)```


**Expected Output*:


```
Pooled output shape: torch.Size([1, 32, 13, 13])```


Here, the spatial dimensions are halved rom 26× 26 to 13× 13 due to the pooling operation.

---

### 🧱 Building a Simple CNN for MNIST

Combining the convolutional and pooling layers, here's a simple CNN moel:


```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # Output: (32, 26, 26)
        self.pool = nn.MaxPool2d(2, 2)    # Output: (32, 13, 13)
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 10)     # 10 output classes for digits 0-9

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32 * 13 * 13)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return 
```


This model consists of:

- A convolutional layer followed by ReLU activation and max pooling.
- A fully connected layer with 128 units.
- An output layer with 10 units corresponding to the digit classes.

---

Feel free to experiment with different configurations, such as adding more convolutional layers, changing kernel sizes, or incorporating dropout for regularization. If you have further questions or need assistance with training this model, let me know! 