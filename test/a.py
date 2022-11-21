#%%
from collections import OrderedDict
import torch
import timm

#%%
print('hello')
#%%
# print(timm.list_models(pretrained=True))


#%%
model = timm.create_model('vgg16', pretrained=True)
print(model)
print('\n\n')
print(model.head)

#%%

for param in model.parameters():
    param.requires_grad = False

#%% FREEZE MODEL

for param in model.features.parameters():
    print(param.requires_grad)
    # freeze:
    param.requires_grad = False

#%% GET LAYERS in features
for layer in model.features.modules():
    print(layer)

#%% Change layer
model.features[1] = torch.nn.Conv2d(64, 64, kernel_size=1)

for layer in model.features.modules():
    print(layer)

#%% CHange layer 2
model.pre_logits = torch.nn.Sequential(
                        torch.nn.Conv2d(512, 1024, kernel_size=3, stride=2),
                        torch.nn.LeakyReLU(),
                        torch.nn.Dropout(p=0.25),
                        torch.nn.Conv2d(1024, 4096, kernel_size=3),
                        torch.nn.LeakyReLU()
)   # layer name will be 0, 1, 2, etc.

model.head = torch.nn.ModuleDict({
            'Q' : torch.nn.Conv2d(512, 1024, kernel_size=3, stride=2),
            'U' : torch.nn.LeakyReLU(),
            'A' : torch.nn.Dropout(p=0.25),
            'N' : torch.nn.Conv2d(1024, 4096, kernel_size=3),
            'H' : torch.nn.LeakyReLU()
})

for layer in model.modules():
    print(layer)

#%%

class MyModel(torch.nn.Module):
    def __init__(self) -> None:
        super(MyModel, self).__init__()
        self.blck = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=3, out_channels=16,
                            kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(in_features=16*28*28, out_features=1)
        )
        # self.blck = torch.nn.Sequential(OrderedDict({
        #         'conv1' : torch.nn.Conv2d(in_channels=3, out_channels=16,
        #                     kernel_size=3, stride=1, padding=1),
        #         'flat1' : torch.nn.Flatten(),
        #         'linear1' : torch.nn.Linear(in_features=16*28*28, out_features=1)
        # }))


test_model = MyModel()
test_model