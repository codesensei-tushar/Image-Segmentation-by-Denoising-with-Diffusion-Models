import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from dataloader import healthy_loader, tumor_loader
from model import UNet, DDPM
import matplotlib.pyplot as plt

def train_model(
    num_epochs,
    learning_rate,
    device=torch.device("cuda")
):
    # Initialize model and diffusion process
    model = UNet().to(device)
    ddpm = DDPM()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training metrics
    train_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        # Train on healthy images only
        progress_bar = tqdm(healthy_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            images = batch['image'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Randomly sample timesteps
            t = torch.randint(0, ddpm.steps, (images.shape[0],)).to(device)
            
            # Add noise to images
            noisy_images, noise = ddpm.add_noise(images, t)
            
            # Predict noise
            predicted_noise = model(noisy_images, t)
            
            # Calculate loss
            loss = criterion(predicted_noise, noise)
            
            # Backpropagate and optimize
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            progress_bar.set_postfix({'loss': sum(epoch_losses) / len(epoch_losses)})
            
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_loss)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoints/model_epoch_{epoch+1}.pth')
            
    return model, train_losses

def evaluate_model(
    model,
    ddpm,
    test_loader,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    threshold=0.1
):
    model.eval()
    results = {
        'true_positives': 0,
        'false_positives': 0,
        'true_negatives': 0,
        'false_negatives': 0
    }
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Generate random timesteps
            t = torch.randint(0, ddpm.steps, (images.shape[0],)).to(device)
            
            # Add noise and try to denoise
            noisy_images, _ = ddpm.add_noise(images, t)
            denoised_images = model(noisy_images, t)
            
            # Calculate reconstruction error
            errors = nn.MSELoss(reduction='none')(denoised_images, images).mean([1,2,3])
            
            # Classify based on threshold
            predictions = (errors > threshold).float()
            
            # Update metrics
            for pred, label in zip(predictions, labels):
                if label == 1:  # Tumor
                    if pred == 1:
                        results['true_positives'] += 1
                    else:
                        results['false_negatives'] += 1
                else:  # Healthy
                    if pred == 1:
                        results['false_positives'] += 1
                    else:
                        results['true_negatives'] += 1
    
    # Calculate metrics
    total = sum(results.values())
    accuracy = (results['true_positives'] + results['true_negatives']) / total
    precision = results['true_positives'] / (results['true_positives'] + results['false_positives'])
    recall = results['true_positives'] / (results['true_positives'] + results['false_negatives'])
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'raw_results': results
    }

if __name__ == "__main__":
    # Create checkpoint directory
    import os
    os.makedirs('checkpoints', exist_ok=True)
    
    # Train model
    print("Starting training...")
    model, losses = train_model()
    
    # Plot training losses
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    
    # Evaluate on tumor dataset
    print("\nEvaluating model...")
    ddpm = DDPM()
    metrics = evaluate_model(model, ddpm, tumor_loader)
    
    print("\nResults:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")