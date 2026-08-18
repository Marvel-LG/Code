from matplotlib import pyplot as plt
import torch
def Count_and_Value(data, name):
    x = []
    y = []
    for epoch, value in data:
        x.append(epoch)
        y.append(value if not torch.is_tensor(value) else value.detach().cpu().numpy())
    plt.plot(x, y, color='red')
    plt.xlabel('epoch')
    plt.ylabel(name[-11:])
    plt.savefig(f'{name}.png')
    plt.close()