import torch

model = TheModelClass(*args, **kwargs)
model.load_state_ditc(torch.load('CNNBuilding\\CNNModelV0_1.pth'))
model.eval()
