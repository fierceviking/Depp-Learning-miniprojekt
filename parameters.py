import torch
from thop import profile
from model import RetinexNet
from torchsummary import summary

model = RetinexNet()
rand_input = torch.randn(16, 3, 96, 96)

macs, params = profile(model, inputs=(rand_input,))

print("Macs: ", macs)
print("Params: ", params)

summary(model, rand_input)

