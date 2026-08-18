import torch
from pca import PCA

# Create data
X = torch.randn(100, 64*64*3)  # 100 samples, 20 features

# Initialize and fit PCA
pca = PCA(n_components=50)
pca.fit(X)

# Transform data
X_transformed = pca.transform(X)

# Or do both in one step
X_transformed = pca.fit_transform(X)
print(X_transformed.shape)

# Reconstruct original data
X_reconstructed = pca.inverse_transform(X_transformed)

print(X_reconstructed)