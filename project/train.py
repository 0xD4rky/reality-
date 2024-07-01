from vis import *
from arch import *

from tqdm.auto import tqdm
import time

def train(model,train_loader,val_loader,criterion,optimizer,num_epochs,save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    total_train_time = 0.0
    best_val_acc     = 0.0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total   = 0
        
        train_pbar = tqdm(train_loader,desc = f"Epoch {epoch+1}/{num_epochs} [Train]")
        for X,y in train_pbar:
            X,y = X.to(device) , y.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred,y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = y_pred.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
            train_pbar.set_postfix({'loss' : f"{running_loss/len(train_loader):.4f}", 'acc' : f"{correct/total:.4f}"})
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct/total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total   = 0
        
        val_pbar = tqdm(val_loader, desc = f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for X,y in val_pbar:
                X,y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = criterion(y_pred, y)
                val_loss += loss.item()
                _, predicted = y_pred.max(1)
                val_total   += y.size(0)
                val_correct += predicted.eq(y).sum().item
                
                val_pbar.set_postfix({'loss': f"{val_loss/len(val_loader):.4f}", 'acc': f"{val_correct/val_total:.4f}"})
                
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        epoch_time = time.time() - start_time
        total_train_time += epoch_time
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with validation accuracy: {best_val_acc:.4f}")
    
    print(f"Total training time: {total_train_time:.2f} seconds")
    return best_val_acc

def predict_image(model, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = output.max(1)
    
    return "Real" if predicted.item() == 1 else "Fake"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
  
train_dataset = Data(root_dir=r'C:\Users\DELL\OneDrive\Documents\reality\data\train', transform=transform)
val_dataset = Data(root_dir=r'C:\Users\DELL\OneDrive\Documents\reality\data\val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = Deepfake()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
  
save_path = 'best_image_classifier.pth'
start_time = time.time()
best_accuracy = train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, save_path=save_path)
total_time = time.time() - start_time

print(f"Training completed. Best validation accuracy: {best_accuracy:.4f}")
print(f"Total execution time: {total_time:.2f} seconds")

best_model = Deepfake()
best_model.load_state_dict(torch.load(save_path))