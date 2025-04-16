import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, lattice_size=20):
        super(Discriminator, self).__init__()
        
        self.input_height = lattice_size
        self.input_width = lattice_size
        
        # Calculate sizes after convolutions
        # First conv: size = (input_size - kernel_size + 2*padding)/stride + 1
        h1 = (lattice_size - 5 + 2*2)//2 + 1  # After conv1
        w1 = (lattice_size - 5 + 2*2)//2 + 1
        
        h2 = (h1 - 5 + 2*2)//1 + 1  # After conv2
        w2 = (w1 - 5 + 2*2)//1 + 1
        
        self.final_height = h2
        self.final_width = w2
        
        # Image processing layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 5, stride = 2, padding = 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 5, stride = 1, padding = 2)
        
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Label processing layers
        self.fc1 = nn.Linear(17, 1024)
        self.fc2 = nn.Linear(1024, 10 * 10 * 128)
        self.batch_norm = nn.BatchNorm1d(10 * 10 * 128)
        
        # Final classifier layers
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear((self.final_height * self.final_width * 32) + (10 * 10 * 128), 1024)
        
        self.fc4 = nn.Linear(1024, 1)

    def forward(self, img, labels):
        # Process image
        x = self.conv1(img)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # Process labels
        y = self.leaky_relu(self.fc1(labels))
        y = self.leaky_relu(self.fc2(y))
        y = self.batch_norm(y)
        y = y.view(-1, 10, 10, 128)
        
        # Concatenate image and label embeddings
        x = x.view(-1, self.final_height * self.final_width * 32)
        y = y.view(-1, 10 * 10 * 128)
        xy = torch.cat((x, y), dim=1)
        
        xy = self.leaky_relu(self.fc3(xy))
        xy_final = self.fc4(xy)
        
        return xy_final

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Noise input (x)
        self.x_dense = nn.Linear(100, 10 * 10 * 256, bias=False)
        self.x_bn = nn.BatchNorm1d(10 * 10 * 256)
        self.x_leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        # Label input (y)
        self.y_dense = nn.Linear(17, 10 * 10 * 128, bias=False)
        self.y_bn = nn.BatchNorm1d(10 * 10 * 128)
        self.y_leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        # Convolutional layers
        self.conv1 = nn.ConvTranspose2d(384, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2, bias=False)

        self.tanh = nn.Tanh()

    def forward(self, x, y):
        # Process noise input
        x = self.x_dense(x)
        x = self.x_bn(x)
        x = self.x_leaky_relu(x)
        x = x.view(-1, 256, 10, 10)

        # Process label input
        y = self.y_dense(y)
        y = self.y_bn(y)
        y = self.y_leaky_relu(y)
        y = y.view(-1, 128, 10, 10)

        # Concatenate along channel dimension
        xy = torch.cat([x, y], dim=1)

        # Apply transpose convolutions
        xy = self.conv1(xy)
        xy = self.bn1(xy)
        xy = torch.relu(xy)

        xy = self.conv2(xy)
        xy = self.bn2(xy)
        xy = torch.relu(xy)

        xy = self.conv3(xy)
        xy = self.bn3(xy)
        xy = torch.relu(xy)

        xy = self.conv4(xy)
        xy = self.tanh(xy)

        return xy

