from utils import sort_dataset
import numpy as np
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import os
import imageio
from PIL import Image
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

@torch.no_grad()
def visualize(data,model,save_path='',epoch=0,num_classes=7):
    model.eval()
    X = model(data.x,data.edge_index).cpu().numpy()
    heatmap_path = os.path.join(save_path,f'heatmap_{epoch}.png')
    sampled_x,_ = sort_dataset(X,data.y,num_classes,stack=True)
    # sim = cosine_similarity(sampled_x,sampled_x)
    # plot_heatmap(sim,heatmap_path)
    # tsne_visualize(X,data.y.cpu(),save_path,epoch,num_classes)
    PCA_visualize(X,data.y.cpu(),save_path,epoch,num_classes)
    # PCA_between_classes(X,data.y.cpu().numpy(),save_path,epoch,num_classes)




    
    
def plot_heatmap(X,save_path):
    plt.rc('text', usetex=False)
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    im = ax.imshow(X, cmap='Blues')
    fig.tight_layout()
    plt.xticks([])
    plt.yticks([])
    fig.savefig(save_path)

    plt.clf()
    plt.close()
    
    
def tsne_visualize(X,Y,save_path='',epoch=0,num_classes=7):
    save_path = os.path.join(save_path,f'tsne_{epoch}.png')
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    plt.scatter(tsne_results[:,0],tsne_results[:,1],c=Y,cmap='tab10',s=5)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.xticks(fontname="Times New Roman", fontsize=15,fontweight='bold')
    plt.yticks(fontname="Times New Roman", fontsize=15,fontweight='bold')
    plt.clf()
    plt.close()
    
    
def PCA_visualize(X,Y,save_path='',epoch=0,num_classes=7):
    save_path = os.path.join(save_path,f'pca_{epoch}.png')
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    plt.scatter(pca_results[:,0],pca_results[:,1],c=Y,cmap='tab10',s=5)
    plt.xticks(fontname="Times New Roman", fontsize=15,fontweight='bold')
    plt.yticks(fontname="Times New Roman", fontsize=15,fontweight='bold')
    ax.set_ylim(-0.1,0.15)
    ax.set_xlim(-0.06,0.20)
    fig.tight_layout()
    fig.savefig(save_path)
    
    plt.clf()
    plt.close()    
    
    
def PCA_between_classes(X,Y,save_path='',epoch=0,num_classes=7):
    cmaps = plt.get_cmap('tab10',num_classes)
    colors = cmaps(np.linspace(0,1,num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            if i < j:
                X_temp = X[(Y==i)|(Y==j)]
                Y_temp = Y[(Y==i)|(Y==j)]
                new_colors = np.array([cmaps.colors[i],cmaps.colors[j]])
                new_cmap = LinearSegmentedColormap.from_list('custom_cmap', new_colors, N=2)
                save_path_temp = os.path.join(save_path,f'class_{i}{j}.png')
                pca_results = PCA(n_components=2).fit_transform(X_temp)
                fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True, dpi=400)
                plt.scatter(pca_results[:,0],pca_results[:,1],c=Y_temp,cmap=new_cmap,s=5)
                plt.title(f'Class {i} and Class {j}',fontsize=15,fontname="Times New Roman",fontweight='bold')
                plt.xticks(fontname="Times New Roman", fontsize=15,fontweight='bold')
                plt.yticks(fontname="Times New Roman", fontsize=15,fontweight='bold')
                fig.tight_layout()
                fig.savefig(save_path_temp)
                plt.clf()
                plt.close()    
            