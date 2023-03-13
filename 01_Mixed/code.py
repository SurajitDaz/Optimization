import torch
import torchvision
import numpy as np
import torchvision.transforms as t
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torchvision.models as model
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

snapshot_path = '/content/drive/MyDrive/Educazone/Task1'
plot_path = snapshot_path

batch_s = 50
num_epoch = 300
learning_rate = 0.001
momentum = 0.9
transform=t.Compose([t.Resize((28,28)),
                     t.ToTensor()])


train_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=True, 
                                          transform=transform,  
                                          download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transform) 

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                          batch_size=batch_s, 
                                          shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_s, 
                                          shuffle=False)
class neuralNetwork(nn.Module):
  def __init__(self, im_h = 28, im_w = 28, num_neurones = 200):
    super(neuralNetwork, self).__init__()
    self.layer1 = nn.Linear(im_h*im_w, num_neurones)
    self.clf = nn.Sequential(nn.Linear(num_neurones, 10),
                             nn.Softmax())
  def forward(self, x):
    #out = x.reshape(x.size(0), -1)
    out = torch.flatten(x, 1)
    out = self.layer1(out)
    out = self.clf(out)
    return out


net = neuralNetwork()
net=net.cuda()
criterion=nn.CrossEntropyLoss()
params = net.parameters()
optimizer=torch.optim.SGD(net.parameters(), lr = learning_rate, momentum=momentum)

model_name = 'neuralNetwork'

load_model=snapshot_path+'/model_'+model_name+'.pth'
loaded_flag=False
if os.path.exists(load_model):
    checkpoint=torch.load(load_model)
    net.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    print("model loaded successfully")
    print('starting training after epoch: ',checkpoint['epoch'])
    loaded_flag=True
    

def plot(val_loss,train_loss):
    plt.title("Loss after epoch: {}".format(len(train_loss)))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(list(range(len(train_loss))),train_loss,color="r",label="Train_loss")
    plt.plot(list(range(len(val_loss))),val_loss,color="b",label="Validation_loss")
    plt.legend()
    path = os.path.join(plot_path,"loss_"+model_name+".png")
    print(path)
    plt.savefig(path)
    #plt.figure()
    plt.close()



val_interval=1
max_acc = 0.0
min_loss=99999
val_loss_gph=[]
train_loss_gph=[]


if loaded_flag:
    min_loss=checkpoint['loss']
    val_loss_gph=checkpoint["val_graph"]
    train_loss_gph=checkpoint["train_graph"]


for epoch in range(num_epoch):
    print("Epoch {}".format(epoch+1))
    train_loss=0.0
    correct=total=0
    val_loss=0
    correct_val=total_val=0
    #net = net.train()
    for i, (image,label) in enumerate(train_loader):
      if batch_s * i < 3200 :
        #print("Epoch {} | batch : {} | total : {}".format(epoch+1, i+1, len(train_loader)))
        net.train()
        optimizer.zero_grad()
        outputs2=net(image.cuda())
        #data1.append(outputs1)
        loss=criterion(outputs2 ,label.cuda())
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()*image.size(0)
        _, predicted = torch.max(outputs2.data, 1)
        total += label.size(0)
        correct += (predicted == label.cuda()).sum().item()

      elif batch_s*i<4000 and batch_s*i>= 3200:
        net.eval()
        with torch.no_grad():
          #for (img_v,lab_v ) in val_loader:
          lab_v = label
          img_v = image
          output_v2=net(image.cuda())
          val_loss+=criterion(output_v2,label.cuda())*img_v.size(0)
          _, predicted = torch.max(output_v2.data, 1)
          total_val += lab_v.size(0)
          correct_val += (predicted == lab_v.cuda()).sum().item()
        
      if min_loss>val_loss:
        state={
            "epoch":i if not loaded_flag else i+checkpoint['epoch'],
            "model_state":net.cpu().state_dict(),
            "optimizer_state":optimizer.state_dict(),
            "loss":min_loss,
            "train_graph":train_loss_gph,
            "val_graph":val_loss_gph,
        }
            
      min_loss=val_loss
      torch.save(state,os.path.join(snapshot_path,"model_"+model_name+'.pt'))
      net.cuda()
    print("Train accuracy", (100*correct/total))
    train_loss_gph.append(train_loss/(3200))
    print("Val accuracy", (100*correct/total))
    val_acc = 100*correct/total
    val_loss_gph.append(val_loss/800)
    print("validation loss : {:.6f} ".format(val_loss/800))
    plot(val_loss_gph, train_loss_gph)


confusion_matrix = torch.zeros(10, 10)

net=net.eval()
correct = 0
total = 0
with torch.no_grad():
      for i, data in enumerate(test_loader):
          if i*batch_s >= 1000:
            break
          images, labels = data
          labels=labels.cuda()
          outputs2 = net(images.cuda())
          _, predicted = torch.max(outputs2.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
          for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print('Accuracy of the network on the test images: %d %%' % (
      100 * correct / total))
print(confusion_matrix)

cm_df = pd.DataFrame(confusion_matrix)
cm_df.to_csv(snapshot_path+'/confusion_matrix.csv')
