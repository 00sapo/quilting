import numpy as np
from tqdm import trange

from ._quilting import transfer


def quilt(
    texture: np.ndarray,
    target: np.ndarray,
    patchLength: int,
    n: int,
    overlap: int = None,
) -> np.ndarray:
    """
    Iteratively applies the transfer function to perform multi-scale synthesis.

    This function progressively refines the synthesis result by iterating over the `transfer` function.
    In each iteration, the alpha blending parameter is adjusted, and the patch length is modified to
    create different levels of detail. The results from previous iterations are used as a prior to inform
    the next level of synthesis.

    Args:
    - texture: The source texture image as a numpy array.
    - target: The target image where the texture will be transferred.
    - patchLength: The initial size of the square patch side.
    - n: The number of synthesis iterations to perform.

    Returns:
    - A numpy array of the target image with the final transferred texture, after n iterations.
    """
    import time

    ttt = time.time()
    res = transfer(texture, target, patchLength)
    print("time taken for 1st transfer", time.time() - ttt)
    for i in trange(1, n, desc="Quilting..."):
        alpha = 0.1 + 0.8 * i / (n - 1)
        patchLength = patchLength * 2**i // 3**i
        res = transfer(
            texture,
            target,
            patchLength,
            alpha=alpha,
            level=i,
            prior=res,
            mode="cut",
            blur=True,
            overlap=overlap,
        )

    return res
