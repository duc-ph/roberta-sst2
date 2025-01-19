import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def get_loss_and_accuracy(model, dataloader, device, eval_ratio=1.0):
    final_accuracy = 0
    final_loss = 0
    new_dataloader = DataLoader(
        dataloader.dataset, 
        batch_size=dataloader.batch_size, 
        shuffle=True, # turn this on
        num_workers=dataloader.num_workers, 
        drop_last=dataloader.drop_last
    )
    num_batch = int(len(new_dataloader) * eval_ratio)

    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(new_dataloader):
            if batch_idx == num_batch:
                break
            logits = model(**x).logits
            softmax = F.softmax(logits, dim=1)
            loss = F.cross_entropy(logits, y)
            y_preds = torch.argmax(softmax, dim=1)
            y = torch.Tensor([x[1] for x in y]).to(device)

            accuracy = (torch.sum(y_preds == y)/len(y)).item()
            final_accuracy += accuracy
            final_loss += loss.item()
    
    final_accuracy /= len(dataloader)
    final_loss /= len(dataloader)
    return loss, final_accuracy