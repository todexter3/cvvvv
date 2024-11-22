import torch
import torch.nn as nn
import torch.optim as optim
import cv2

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=5, padding=2)
        # self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4096, num_classes)
        self.fc2 = nn.Linear(num_classes, num_classes)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_data():
    '''
    Load the data from the disk
    '''
    # TODO
    data_img = cv2.resize(cv2.imread(f'images/fur_obs.jpg', cv2.IMREAD_GRAYSCALE), (64, 64))
    data_img = data_img * 7 // 255
    data_img = torch.from_numpy(data_img)
    # flatten the data_img
    data_img = data_img.view(-1, 1, 64, 64)

    # turn the image into a 1-dimensional vector
    label = data_img.view(-1, 64 * 64)
    return [data_img, label]


def run():
    data, label = load_data()
    print(data.shape)
    print(label.shape)

    # Example usage:
    input_channels = 1  # Example number of input channels for grayscale images
    num_classes = 4096  # Example number of classes for classification

    # Create an instance of the SimpleCNN model
    cnn_model = SimpleCNN(input_channels, num_classes)

    # Define loss function and optimizer
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    data = torch.tensor(data, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.float32)
    # train the model with data and label
    for epoch in range(20):
        running_loss = 0.0
        optimizer.zero_grad()
        outputs = cnn_model(data)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(f"epoch {epoch}, training loss is: {running_loss}")
        running_loss = 0.0

    # Save the model
    torch.save(cnn_model.state_dict(), 'model.pth')

    # load the model
    model = SimpleCNN(input_channels, num_classes)
    model.load_state_dict(torch.load('model.pth'))
    
    outputs = model(data)
    # reshape the output as 64*64
    outputs = outputs.view(64, 64)
    outputs = outputs.detach().numpy()

    # save the output as a image
    cv2.imwrite("output.jpg", outputs)


def get_CNN_filters():
    # load the model
    model = SimpleCNN(1, 4096)
    model.load_state_dict(torch.load('model.pth'))
    # get the parameters of the model
    params = model.state_dict()
    # get the convolution layers of the model
    conv1 = params['conv1.weight']
    conv2 = params['conv2.weight']

    # Turn them into numpy array
    conv1 = conv1.detach().numpy()
    conv2 = conv2.detach().numpy()

    conv1 = conv1.tolist()
    conv2 = conv2.tolist()
    return conv1, conv2

def test_the_model():
    data, label = load_data()
    data = torch.tensor(data, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.float32)
    # Example usage:
    input_channels = 1  # Example number of input channels for grayscale images
    num_classes = 4096  # Example number of classes for classification
    # load the model
    model = SimpleCNN(input_channels, num_classes)
    model.load_state_dict(torch.load('model.pth'))
    
    outputs = model(data)
    # reshape the output as 64*64
    outputs = outputs.view(64, 64)
    outputs = outputs.detach().numpy()

    data_img = cv2.resize(cv2.imread(f'images/fur_obs.jpg', cv2.IMREAD_GRAYSCALE), (64, 64))
    data_img = data_img * 7 // 255
    print(data_img)
    # show the outputs
    print(outputs)


if __name__ == '__main__':
    test_the_model()
    # run()