# quilting

Code adapted and optimized from by Alex Xu: https://github.com/axu2/image-quilting

Install with `pip install git+https://github.com/00sapo/quilting`.

You need a C++ compiler for installing.

Compile in-place for testing: `pdm compile`.

This is only a draft and not carefully tested. Use at your own risk. Read the code. It
spawned from old code in
[AutoDocAugment](https://github.com/LaudareProject/AutoDocAugment) and tested only
there.

It uses multithreading and C++ optimizations for speed.

```python
from quilting import quilt
quilted_image = quilt(
    texture=img0, # texture image to sample from, a numpy array of uint8
    target=img1, # target image to reconstruct, a numpy array of uint8
    patchLength=20, # patch size
    n=2, # number of iterations
    overlap=0 # if 0, defaults to patchLength // 6
)
```

## References:

A. A. Efros and W. T. Freeman, “Image quilting for texture synthesis and transfer,” in Proceedings of the 28th annual conference on Computer graphics and interactive techniques, in SIGGRAPH ’01. New York, NY, USA: Association for Computing Machinery, 2001, pp. 341–346. doi: 10.1145/383259.383296.
