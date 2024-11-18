import torch
from thop import profile
from model import RetinexNet
from torchsummary import summary

model = RetinexNet()
rand_input = torch.randn(1, 3, 256, 256)

macs, params = profile(model, inputs=(rand_input,))

print("Macs: ", macs)
print("Params: ", params)

summary(model, rand_input)

#Macs:  21730738176.0
#Params:  555205.0