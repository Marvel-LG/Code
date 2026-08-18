import numpy as np
import torch
import json
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader
from models.VAE.autoencoder.autoencoder import AautoencoderKL as Autoencoder
from data.dataset import InpaintDataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

json_str = ''
with open("config/inpainting_voc2012.json", 'r') as f:
    for line in f:
        line = line.split('//')[0] + '\n'
        json_str += line
opt = json.loads(json_str, object_pairs_hook=OrderedDict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder_A = Autoencoder(**opt['model']['which_networks'][0]['args']['model_parameter']['unet_A']).to(
    device)
encoder_A.load_state_dict(torch.load('checkpoint/A_model/latest_Network_A.pth'), strict=True)

dataset = InpaintDataset('datasets', image_size=[256, 256], pattern='A', phase='train')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)


def save_tensor_and_labels(tensor_array, labels, filename):
    tensor_array_np = tensor_array.cpu().numpy()
    labels_np = labels.cpu().numpy()

    np.savez_compressed(filename, data=tensor_array_np, labels=labels_np)
    print(f'Saved tensors and labels to {filename}')


def load_array_and_labels(filename):
    data = np.load(filename)
    array = data['data']
    labels = data['labels']
    return array, labels


def normal_encode():
    # from utils.showImage import Show
    latents_array = []
    labels_array = []
    for data in tqdm(dataloader, desc="Processing"):
        inputs = data['synthetic'].to(device)
        labels = data['inst'].to(device)

        with torch.no_grad():
            posterior = encoder_A.encodeing(inputs, labels)
            latents = posterior.sample(inference=True)
        latents_array.append(latents.cpu())
        labels_array.append(data['inst'].cpu())
        # Show(inputs)


    latents_array = torch.cat(latents_array).numpy()
    labels_array = torch.cat(labels_array).numpy()
    return latents_array, labels_array


def unnormal_encode():
    # from utils.showImage import Show
    latents_array = []
    labels_array = []
    for data in tqdm(dataloader, desc="Processing"):
        inputs = data['synthetic'].to(device)
        labels = data['inst'].to(device)
        indices_s = torch.nonzero(labels == 1)
        index_s = indices_s[-1]

        indices_x = torch.nonzero(labels == 2)
        index_x = indices_x[-1]

        labels[index_x] = 0

        inputs[index_x] = inputs[index_s]

        # inputs = torch.cat((inputs, inputs[indices_s]), dim=0)

        # labels = torch.cat((labels, torch.tensor([0, 0]).to(labels)))

        with torch.no_grad():
            posterior = encoder_A.encodeing(inputs, labels)
            latents = posterior.sample(inference=True)
        latents_array.append(latents.cpu())
        labels_array.append(torch.tensor([0, 1, 2, 3]).cpu())
        # Show(inputs, name=labels)

    latents_array = torch.cat(latents_array).numpy()
    labels_array = torch.cat(labels_array).numpy()
    return latents_array, labels_array


def visualize_tsne(data, labels, label_mapping, n_pca_components=0.95, perplexity=30, learning_rate=200):
    b, c, h, w = data.shape
    data_flattened = data.reshape(b, -1)


    pca = PCA(n_components=n_pca_components, whiten=True)
    data_pca = pca.fit_transform(data_flattened)


    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
    data_tsne = tsne.fit_transform(data_pca)


    new_labels = [label_mapping[label] for label in labels]


    plt.figure(figsize=(10, 8))


    shapes = ['o', 's', '^', 'D']
    colors = ['red', 'blue', 'green', 'black']
    alphas = [0.5, 0.5, 0.5, 0.5]
    unique_labels, indices = np.unique(new_labels, return_index=True)
    unique_labels = unique_labels[np.argsort(indices)]
    for i, label in enumerate(unique_labels):
        indices = (np.array(new_labels) == label)
        plt.scatter(data_tsne[indices, 0], data_tsne[indices, 1], marker=shapes[i % len(shapes)], label=label,
                    color=colors[i % len(colors)],
                    alpha=alphas[i % len(alphas)])

    plt.colorbar(label='Labels')
    plt.title(f't-SNE {n_pca_components} {perplexity} {learning_rate}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title="Labels")
    plt.show()


def visualize_umap(data, labels, label_mapping, n_pca_components=1000):
    b, c, h, w = data.shape
    data_flattened = data.reshape(b, -1)

    # 使用 PCA 降维
    pca = PCA(n_components=n_pca_components)
    data_pca = pca.fit_transform(data_flattened)

    # 使用 UMAP 降维到 2D
    umap_model = umap.UMAP(n_components=2, n_neighbors=15)
    data_umap = umap_model.fit_transform(data_pca)


    new_labels = [label_mapping[label] for label in labels]


    plt.figure(figsize=(10, 8))


    shapes = ['o', 's', '^', 'D']
    colors = ['red', 'blue', 'green', 'black']
    alphas = [0.5, 0.5, 0.5, 0.5]
    unique_labels, indices = np.unique(new_labels, return_index=True)
    unique_labels = unique_labels[np.argsort(indices)]

    for i, label in enumerate(unique_labels):
        indices = (np.array(new_labels) == label)
        plt.scatter(data_umap[indices, 0], data_umap[indices, 1], marker=shapes[i % len(shapes)], label=label,
                    color=colors[i % len(colors)],
                    alpha=alphas[i % len(alphas)])

    plt.colorbar(label='Labels')
    plt.title('UMAP Visualization')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.legend(title="Labels")
    plt.show()


def show_input_and_outout():
    from utils.showImage import Show
    for index, data in enumerate(tqdm(dataloader, desc="Processing")):
        inputs = data['synthetic'].to(device)
        labels = data['inst'].to(device)
        labels = torch.ones_like(labels)
        with torch.no_grad():
            out_put = encoder_A(inputs, labels)
        Show(torch.cat((inputs, out_put), dim=0), name=index)



if __name__ == "__main__":
    # show_input_and_outout()

    array, labels = normal_encode()
    label_mapping = {
        0: "Real Old image",
        1: "Synthetic image",
        2: "Clear image",

    }

    pca_components = [0.85, 0.90, 0.95]
    perplexities = [30, 50, 70]
    learning_rates = [100, 200, 300]
    for n_pca_components in pca_components:
        for perplexity in perplexities:
            for learning_rate in learning_rates:  # n_pca_components=0.95, perplexity=30, learning_rate=200
                visualize_tsne(array, labels, label_mapping, n_pca_components=n_pca_components, perplexity=perplexity,
                               learning_rate=learning_rate)
    # visualize_umap(array, labels, label_mapping)

    array, labels = unnormal_encode()
    label_mapping = {
        0: "Real Old image",
        1: "Synthetic image",
        2: "Clear image",
        3: "Fake Synthetic image",
    }

    pca_components = [0.85, 0.90, 0.95]
    perplexities = [30, 50, 70]
    learning_rates = [200, 300,500]
    for n_pca_components in pca_components:
        for perplexity in perplexities:
            for learning_rate in learning_rates:  # n_pca_components=0.95, perplexity=30, learning_rate=200
                visualize_tsne(array, labels, label_mapping, n_pca_components=n_pca_components, perplexity=perplexity,
                               learning_rate=learning_rate)
    # visualize_tsne(array, labels, label_mapping)
    # visualize_umap(array, labels, label_mapping)
