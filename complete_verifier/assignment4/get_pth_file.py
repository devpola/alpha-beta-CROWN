import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 모델 정의 (이전에 작성한 fashion_mnist_model)
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.model = nn.Sequential(
          nn.Conv2d(1, 16, 3, stride=1, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2),
          nn.Conv2d(16, 32, 3, stride=1, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2),
          nn.Flatten(),
          nn.Linear(32 * 7 * 7, 128),  # FC Layer
          nn.ReLU(),
          nn.Linear(128, 10)          # Output Layer
      )

    def forward(self, x):
        return self.model(x)

# 하이퍼파라미터 설정
batch_size = 64
learning_rate = 0.001
epochs = 3

# 데이터셋 및 데이터로더 설정
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.286], std=[0.353])  # Fashion MNIST 통계값
])

train_dataset = datasets.FashionMNIST(root='datasets', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='datasets', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델, 손실 함수, 최적화기 정의
model = FashionMNISTModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 루프
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 순전파
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 테스트 데이터로 모델 평가
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test data: {100 * correct / total:.2f}%")

# 모델 가중치 저장
sd = model.state_dict()
new_sd = {}
for k, v in sd.items():
    new_key = k.replace("model.", "")  # "model." 접두사를 제거
    new_sd[new_key] = v
torch.save(new_sd, "fashion_mnist_model.pth")
print("Model saved as 'fashion_mnist_model.pth'")
