import torchvision.utils as vutils


def Show(tensor, t=None):
    vutils.save_image((tensor + 1) / 2, f'./experiments/sample/{t}.png')
