import os
import torch
import matplotlib.pyplot as plt
from train import train_model, evaluate_model
from model import DDPM
from dataloader import tumor_loader

os.makedirs('checkpoints', exist_ok=True)
    
# Training parameters (adjust as needed)
num_epochs = 50
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
print("Starting training...")
model, losses = train_model(num_epochs=num_epochs, learning_rate=learning_rate, device=device)

# Plot training loss over time
plt.plot(losses)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('training_loss.png')
plt.show()
    
print("\nEvaluating model...")
ddpm = DDPM()
metrics = evaluate_model(model, ddpm, tumor_loader, device=device)
    
print("\nResults:")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
