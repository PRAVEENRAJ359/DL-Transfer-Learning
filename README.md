# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Developing a Neural Network Classification Model using Transfer Learning

Training deep neural networks from scratch requires large datasets, high computational power, and significant training time. In many practical scenarios, such resources are limited.

The goal of this project is to develop an image classification model using Transfer Learning, where a pre-trained neural network is reused and fine-tuned to classify new data efficiently.

## Neural Network Model
A Neural Network Model is a type of machine learning model inspired by the structure and functioning of the human brain. It is used to learn patterns from data and make predictions or decisions.
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/d43e5227-d911-4186-b071-3b68b96e5d3f" />

## DESIGN STEPS
### STEP 1: 

Import required libraries and define image transforms.

### STEP 2: 

Load training and testing datasets using ImageFolder.

### STEP 3: 

Visualize sample images from the dataset.

### STEP 4: 

Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5: 

Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

### STEP 6: 

Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.

### Name: PRAVEEN RAJ R

### Register Number: 212224230207

```python
# Load Pretrained Model and Modify for Transfer Learning

model=models.vgg19(weights=VGG19_Weights.DEFAULT)

# Modify the final fully connected layer to match the dataset classes

model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)

# Include the Loss function and optimizer
criterion =nn.BCEWithLogitsLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)


# Train the model
def train_model(model, train_loader,test_loader,num_epochs=100):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                val_loss += loss.item()
        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: PRAVEEN RAJ R ")
    print("Register Number: 212224230207")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="1001" height="807" alt="image" src="https://github.com/user-attachments/assets/3773666f-8126-463e-bc5e-62126f58d9fc" />


## Confusion Matrix
<img width="868" height="804" alt="image" src="https://github.com/user-attachments/assets/f25b1776-943c-43c5-b848-d6b96d09e6de" />


## Classification Report
<img width="1342" height="421" alt="image" src="https://github.com/user-attachments/assets/f9dc015b-f77a-4c6a-aaa9-efac0a55edc8" />


### New Sample Data Prediction
<img width="565" height="890" alt="image" src="https://github.com/user-attachments/assets/0b71e24b-c624-42cf-907e-307b63b62db4" />


## RESULT
The image classification model using transfer learning with VGG19 architecture for the given dataset has been executed successfully.
