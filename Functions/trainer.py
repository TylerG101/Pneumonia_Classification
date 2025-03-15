import torch 
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

### Contains train_step(), test_step(), and train() functions



def train_step(model:torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               device:torch.device) -> Tuple[float, float]:

    '''
    Passes a given PyTorch model through 1 epoch of the model training process

    Inputs: 
        model: A PyTorch model
        dataloader: PyTorch dataloader the model will train on
        loss_fn: Loss function to be minimized
        optimizer: Optimizer used to minimize the above loss function
        device: Device for computations to be carred out on

    Output:
        Tuple containing training loss and training accuracy
    '''

    #Put model in training model
    model.train()  

    #Set up loss and accuracy values
    train_loss, train_acc = 0, 0

    #Loop through the dataloaders batches
    for batch, (X, y) in enumerate(dataloader):
        #Put the data on device
        X, y  = X.to(device), y.to(device)

        ###fwd pass
        pred = model(X)

        ###Calculate loss
        loss = loss_fn(pred, y)
        train_loss += loss

        ###Zero gradients
        optimizer.zero_grad()

        ###Perform backpropogation
        loss.backward()

        ###Step optimizer
        optimizer.step()

        ###Calculate and accumulate accuracy across all batches
        y_pred_class = torch.argmax(torch.softmax(pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(pred)

    ###Loss and accuracy per batch
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc







def test_step(model:torch.nn.Module,
              dataloader:torch.utils.data.DataLoader,
              loss_fn:torch.nn.Module,
              device:torch.device) -> Tuple[float, float]:
    
    '''
    Passes a given PyTorch model through 1 epoch of the model testing process

    Inputs: 
        model: A PyTorch model
        dataloader: PyTorch dataloader the model will train on
        loss_fn: Loss function for model evaluation
        device: Device for computations to be carred out on

    Output:
        Tuple containing testing loss and testing accuracy
    '''

    ### Put model in eval mode
    model.eval()

    ###Setup test loss and acc values
    test_loss, test_acc = 0, 0 

    with torch.inference_mode():
        for batch, (X,y) in enumerate(dataloader):
            #Send data to device
            X, y = X.to(device), y.to(device)

            ###fwd pass
            test_pred = model(X)

            ###Loss calculation
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()

            ###Accuracy calculation
            accuracy = torch.argmax(test_pred, dim=1)
            test_acc += ((accuracy == y).sum().item()/len(accuracy))

    ###Loss and accuracy per batch
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc







def train(model:torch.nn.Module,
          train_dataloader:torch.utils.data.DataLoader,
          test_dataloader:torch.utils.data.DataLoader,
          loss_fn:torch.nn.Module,
          optimizer:torch.optim.Optimizer,
          epochs:int,
          device:torch.device) -> Dict[str, List]:
  
  '''
  Passes a given PyTorch model through the train_step() function, followed by the test_step() function for a 
  given number of epochs

  Inputs
    model: PyTorch model to be trained
    train_dataloader: Dataloader instance for the train_step() function
    test_dataloader: Dataloader instance for the test)step() function
    loss_fn: Loss function used for training and testing
    optimizer: Optimizer to minimize loss in train_step()
    epochs: Number of times indicating number of loops through the train() function
    device: Device for calculations to be carried out on

  Output
    A dictionary containing train loss and accuracy, and test loss and accuracy
  '''
    
  ###Create the dictionaries to be filled
  results = {"train_loss" : [],
           "train_acc" : [],
           "test_loss" : [],
           "test_acc" : []
           }
  
  #Loop through training and testing steps for the specified number of epochs
  for epoch in tqdm(range(epochs)):
      train_loss, train_acc = train_step(model=model,
                                         dataloader=train_dataloader,
                                         loss_fn=loss_fn,
                                         optimizer=optimizer,
                                         device=device)
      test_loss, test_acc = test_step(model=model,
                                      dataloader=test_dataloader,
                                      loss_fn=loss_fn,
                                      device=device)
      
      ###Print info
      print(
          f'Epoch: {epoch+1} | '
          f'train_loss: {train_loss:.4f} | '
          f'train_acc: {train_acc:.4f} | '
          f'test_loss: {test_loss:.4f} | '
          f'test_acc: {test_acc:.4f}'
      )

      ###Update results dictionary with just the values
      results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
      results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
      results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
      results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

  return results





