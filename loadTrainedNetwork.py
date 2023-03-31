import torch
import torchvision
import matplotlib.pyplot as plt
import Network
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets

# hyperparams
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5

# initialize the network
continued_network = Network.Net()
continued_optimizer = optim.SGD(continued_network.parameters(), lr=learning_rate, momentum=momentum)

# load trained network model
network_state_dict = torch.load('./savedModel/model.pth')
continued_network.load_state_dict(network_state_dict)

# load trained optimizer
optimizer_state_dict = torch.load('./savedModel/optimizer.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)

# load testing data
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./datasets', train=False, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

# prepare testing data
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

# give the network testing data and get the output
with torch.no_grad():
  output = continued_network(example_data)

# Show images with predictions on a trained network
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])
fig
plt.show()
