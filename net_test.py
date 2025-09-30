from thop import profile
import torch
import time
from net.ISGNet import ISGNet

model = ISGNet().to('cuda')  
 
input = torch.rand(1,3,256,256).to('cuda') 
torch.cuda.synchronize()
model.eval()
time_start = time.time()
_ = model(input,input)
time_end = time.time()
torch.cuda.synchronize()
time_sum = time_end - time_start
print(f"Time: {time_sum}")
n_param = sum([p.nelement() for p in model.parameters()])  
n_paras = f"n_paras: {(n_param/2**20)}M\n"
print(n_paras)
macs, params = profile(model, inputs=(input,input)) 
print(f'FLOPs:{macs/(2**30)}G')
