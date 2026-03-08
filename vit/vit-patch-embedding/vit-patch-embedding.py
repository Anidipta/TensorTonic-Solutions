import numpy as np

def patch_embed(image, patch_size, embed_dim):
    B,H,W,C = image.shape
    patches = image.reshape(B, H//patch_size, patch_size, W//patch_size, patch_size, C)\
                   .transpose(0,1,3,2,4,5)\
                   .reshape(B, -1, patch_size*patch_size*C)
    W = np.random.randn(patch_size*patch_size*C, embed_dim)
    return patches @ W