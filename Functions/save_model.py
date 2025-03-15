
###Contains the function to save a PyTorch model.


import torch
from pathlib import Path

def save_model(model:torch.nn.Module,
               target_dir:str,
               model_name:str):
    """Saves a given PyTorch model to a specified directory.

  Inputs:
    model: A PyTorch model to save.
    target_dir: A directory for the model to be saved to.
    model_name: A filename for the saved model.  Designate
      either ".pth" or ".pt" as the file extension.

  """
    #Create directory for the model to be saved to
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)
    
    #Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should in with '.pt' or '.pth' "
    model_save_path = target_dir_path / model_name


    #Save the model state dict() 
    print(f'[INFO] Saving model to: {model_save_path}')
    torch.save(obj=model.state_dict(),
               f=model_save_path)