from torchvision import transforms, datasets
import torch.optim as optim

from src.mnist_model import MNIST_CNN

from src.training import get_accuracy
from src.utils import device
import torch.nn as nn
import torch
from settings import settings

num_classes = 10
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

full_train_dataset = datasets.MNIST(
    root=settings.data_path,
    train=True,
    download=True,
    transform=transform,
)
# Move dataset to the device
train_data = torch.stack(
    [full_train_dataset[i][0] for i in range(len(full_train_dataset))]
).to(device)
train_labels = torch.tensor(
    [full_train_dataset[i][1] for i in range(len(full_train_dataset))], device=device
)

test_dataset = datasets.MNIST(
    root=settings.data_path,
    train=False,
    download=True,
    transform=transform,
)
test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]).to(
    device
)
test_labels = torch.tensor(
    [test_dataset[i][1] for i in range(len(test_dataset))], device=device
)


# Instantiate the model
model = MNIST_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=float(1e-2))

# Training loop
num_epochs: int = 1000
target_accuracy: float = 0.96
maximum_scramble_distance: float = 1.90

model.set_scramble_distance(0.1, 1)

for epoch in range(num_epochs):
    indices = torch.randperm(len(train_data), device=device)
    batch_size = 128
    for i in range(0, len(train_data), batch_size):
        batch_indices = indices[i : i + batch_size]  # Select batch indices
        inputs, labels = (
            train_data[batch_indices],
            train_labels[batch_indices],
        )

        if epoch == 0:
            break

        # Forward pass
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Assess progress:
    model.clean_mode()
    validation_accuracy: float = get_accuracy(model, test_data, test_labels)
    model.binary_mode()
    bin_validation_accuracy: float = get_accuracy(model, test_data, test_labels)
    model.train_mode()
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Accuracy: {validation_accuracy:.4f}, Bin Accuracy: {bin_validation_accuracy:.4f}"
    )
    if validation_accuracy > target_accuracy:
        ds: float = 0.05
        model.set_scramble_distance(ds, 1)
        print(
            f"Scramble distances: "
            f"{model.layer1.scramble_distance:.2f}, "
            f"{model.layer2.scramble_distance:.2f}, "
            f"{model.layer3.scramble_distance:.2f}, "
            f"{model.layer4.scramble_distance:.2f}"
        )
