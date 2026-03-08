import numpy as np

def prepend_class_token(patches: np.ndarray, embed_dim: int) -> np.ndarray:
    B = patches.shape[0]
    cls_token = np.random.randn(1, 1, embed_dim)
    cls_tokens = np.repeat(cls_token, B, axis=0)
    return np.concatenate([cls_tokens, patches], axis=1)