import torch
import torch.distributed as dist

# Processing upload_feature after recv
def recv(tensor:torch.Tensor, src:int, dst:int, broadcast:bool=False, batch_padding:int=0):
    dist.recv(tensor, src=src)

    if batch_padding > 0:
        recv_tensor = tensor[:tensor.size(0)-batch_padding,:]

    return recv_tensor.detach().requires_grad_(True)


# Processing upload_feature and send
def send(tensor:torch.Tensor, src:int, dst:int, broadcast:bool=False, batch_padding:int=0):
    
    if batch_padding > 0:
        pad_tensor = torch.zeros([batch_padding,tensor.shape[1:]],dtype=torch.float16).to(tensor.device)
        send_tensor = torch.cat(tensor, pad_tensor).detach()

    dist.send(send_tensor, dst=dst)