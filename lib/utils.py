import numpy as np

def tup_distance(node1, node2, mode="Euclidean"):
    """
    mode: "Manhattan", "Euclidean"
    """
    if mode=="Euclidean":
        return ((node1[0]-node2[0])**2+(node1[1]-node2[1])**2)**0.5
    elif mode=="Manhattan":
        return np.abs(node1[0]-node2[0])+np.abs(node1[1]-node2[1])
    else:
        raise ValueError("Unrecognized distance mode: "+mode)

def mat_distance(mat1, mat2, mode="Euclidean"):
    """
    mode: "Manhattan", "Euclidean"
    """
    if mode=="Euclidean":
        return np.sum((mat1-mat2)**2, axis=-1)**0.5
    elif mode=="Manhattan":
        return np.sum(np.abs(mat1-mat2), axis=-1)
    else:
        raise ValueError("Unrecognized distance mode: "+mode)
