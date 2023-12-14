# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False, initializedcheck=False, -profile=True
# distutils: language=c++

import math
import numpy as np
cimport numpy as np

cimport cython

import _heapq
from libc.math cimport ceil, abs
from libc.stdint cimport int64_t
from libc.stdlib cimport malloc, free

from cpython cimport bool
from cpython.ref cimport PyObject
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.util import img_as_float, img_as_ubyte
from cython.parallel import prange

cdef np.float64_t sum_abs_diff_2d(np.float64_t[:, :] a, np.float64_t[:, :] b) noexcept nogil:
    """Compute the sum of squared differences between two 2D matrices"""
    if a.shape[0] != b.shape[0] or a.shape[1] != b.shape[1] or a.shape[2] != b.shape[2]:
        raise ValueError("a and b must have the same shape")
    cdef np.float64_t res = 0
    cdef int i, j
    cdef int h = a.shape[0] # // 2 + 1 # // 2 + 1 for speed up
    cdef int w = a.shape[1] # // 2 + 1 # // 2 + 1 for speed up
    for i in range(h):#, nogil=True):
        for j in range(w):
            res += abs(a[i, j] - b[i, j])
    return res

cdef np.float64_t sum_abs_diff_3d(np.float64_t[:, :, :] a, np.float64_t[:, :, :] b) noexcept nogil:
    """Compute the sum of squared differences between two 3D matrices"""
    if a.shape[0] != b.shape[0] or a.shape[1] != b.shape[1] or a.shape[2] != b.shape[2]:
        raise ValueError("a and b must have the same shape")
    cdef np.float64_t res = 0
    cdef int i, j, k
    cdef int h = a.shape[0] # // 2 + 1 # // 2 + 1 for speed up
    cdef int w = a.shape[1] # // 2 + 1 # // 2 + 1 for speed up
    cdef int c = a.shape[2] # // 2 + 1 # // 2 + 1 for speed up
    for i in range(h):#, nogil=True):
        for j in range(w):
            for k in range(c):
                res += abs(a[i, j, k] - b[i, j, k])
    return res

cdef (int, int) _unravel_index_argmin(np.float64_t[:, :] errors) noexcept:
    """Unravel the index of the minimum value in a 2D array"""
    cdef int i, j
    cdef np.int64_t minIdx = 0
    cdef np.float64_t minVal = errors[0, 0]
    cdef int rows = errors.shape[0]
    cdef int cols = errors.shape[1]
    cdef int x, y
    
    for i in range(rows):
        for j in range(cols):
            if errors[i, j] < minVal:
                minVal = errors[i, j]
                minIdx = i * cols + j

    x = minIdx // cols
    y = minIdx % cols
    return x, y

cdef np.float64_t[:, :, :] _bestCorrPatch(np.float64_t[:, :, :] texture, np.float64_t[:, :] corrTexture, int patchLength, np.float64_t[:, :] corrTarget, int y, int x) noexcept:
    cdef int h = texture.shape[0], w = texture.shape[1]
    cdef np.float64_t[:, :] errors
    errors = np.zeros((h - patchLength + 1, w - patchLength + 1))
    cdef np.float64_t[:, :] corrTargetPatch = corrTarget[y : y + patchLength, x : x + patchLength]
    cdef int curPatchHeight = corrTargetPatch.shape[0], curPatchWidth = corrTargetPatch.shape[1]

    cdef int i, j
    cdef np.float64_t[:, :] corrTexturePatch
    cdef np.float64_t[:, :] e
    cdef np.float64_t[:, :, :] texturePatch
    for i in prange(h - patchLength + 1, nogil=True):
        for j in range(0, w - patchLength + 1):
            errors[i, j] = sum_abs_diff_2d(corrTexture[i : i + patchLength, j : j + patchLength], corrTargetPatch)
    # i, j = np.unravel_index(np.argmin(errors), (errors.shape[0], errors.shape[1]))
    # np.unravel need the gil, so we use the following code instead
    i, j = _unravel_index_argmin(errors)
    return texture[x : x + patchLength, y : y + patchLength, :]

cdef np.float64_t[:, :, :] _bestCorrOverlapPatch(
    np.float64_t[:, :, :] texture,
    np.float64_t[:, :] corrTexture,
    int patchLength,
    int overlap,
    np.float64_t[:, :] corrTarget,
    np.float64_t[:, :, :] res,
    int y,
    int x,
    float alpha=0.1,
    int level=0,
) noexcept :
    cdef int h = texture.shape[0]
    cdef int w = texture.shape[1]
    cdef np.float64_t[:, :] errors
    errors = np.zeros((h - patchLength, w - patchLength))
    cdef np.float64_t[:, :] corrTargetPatch = corrTarget[y : y + patchLength, x : x + patchLength]
    cdef int di = corrTargetPatch.shape[0]
    cdef int dj = corrTargetPatch.shape[1]
    # print(f"corrTargetPatch dimensions: di = {di}, dj = {dj}")
    if di == 0 or dj == 0:
        texture[:di, :dj] # just something with the dimension 0 so we can detect this case from the caller function
    cdef np.float64_t[:, :] corrTexturePatch
    cdef float corrError, prevError, overlapError
    cdef int i, j
    for i in prange(h - patchLength, nogil=True):
        for j in range(0, w - patchLength):
            # print(f"Patch dimensions: {patch.shape}, Location: ({i}, {j})")
            overlapError = _L2OverlapDiff(texture[i : i + di, j : j + dj, :], overlap, res, y, x)
            # print(f"Overlap error: {overlapError}")
            corrError = sum_abs_diff_2d(corrTexture[i : i + di, j : j + dj], corrTargetPatch)
            # print(f"Correspondence error: {corrError}")
            prevError = 0
            if level > 0:
                prevError = sum_abs_diff_3d(texture[i + overlap : i + di, j +overlap : j + dj, :], res[y + overlap : y + patchLength, x + overlap : x + patchLength])
                # print(f"Previous level error: {prevError}")
            errors[i, j] = alpha * (overlapError + prevError) + (1 - alpha) * corrError
            # print(f"Total error at ({i}, {j}): {errors[i, j]}")
    
    # i, j = np.unravel_index(np.argmin(errors), (errors.shape[0], errors.shape[1]))
    i, j = _unravel_index_argmin(errors)
    # print(f"Minimum error found at: ({i}, {j}), with error: {errors[i, j]}")
    return texture[i:i+di, j:j+dj]
    # print(f"Returning patch with dimensions: {ret.shape}")
    # return ret

cdef float _L2OverlapDiff(np.float64_t[:, :, :] patch, int overlap, np.float64_t[:, :, :] res, int y, int x) noexcept nogil:
    cdef float error = 0.0
    if x > 0 and y > 0:
        error = sum_abs_diff_3d(patch[:overlap, :overlap], res[y : y + overlap, x : x + overlap])
    elif y > 0:
        error = sum_abs_diff_3d(patch[:overlap:, :], res[y : y + overlap, x : x + patch.shape[1]])
    elif x > 0:
        error = sum_abs_diff_3d(patch[:, :overlap], res[y : y + patch.shape[0], x : x + overlap])
    return error

cdef _minCutPath(np.float64_t[:, :] errors) noexcept:
    # Extracting the shapes of the input numpy array
    cdef int h = errors.shape[0]
    cdef int w = errors.shape[1]
    
    cdef int i, j, delta, nextIndex, curDepth
    cdef double error, cumError
    cdef list path = []
    cdef list pq = []
    cdef set seen = set()
    
    # Initialize the priority queue with the first row of the errors matrix
    for i in range(w):
        _heapq.heappush(pq, (errors[0, i], [i]))
    
    while pq:
        error, path = _heapq.heappop(pq)
        curDepth = len(path)
        curIndex = path[len(path) - 1]
        
        # When a path reaches the last row, return this path as it will have the least cumulative error
        if curDepth == h:
            return path
        
        # Explore the neighboring pixels in the next row
        for delta in (-1, 0, 1):
            nextIndex = curIndex + delta
            if 0 <= nextIndex < w and (curDepth, nextIndex) not in seen:
                cumError = error + errors[curDepth, nextIndex]
                _heapq.heappush(pq, (cumError, path + [nextIndex]))
                seen.add((curDepth, nextIndex))

cdef np.float64_t[:, :] sum_abs_diff_axis_2(np.float64_t[:, :, :] a, np.float64_t[:, :, :] b) noexcept:
    # sum only over axis 2
    if a.shape[0] != b.shape[0] or a.shape[1] != b.shape[1] or a.shape[2] != b.shape[2]:
        raise ValueError("a and b must have the same shape")
    cdef np.float64_t[:, :] res
    res = np.zeros((a.shape[0], a.shape[1]))
    cdef int i, j, k
    cdef int h = a.shape[0]
    cdef int w = a.shape[1]
    cdef int c = a.shape[2]

    for i in range(h):#,=True):
        for j in range(w):
            for k in range(c):
                res[i, j] += ((a[i, j, k] - b[i, j, k]) ** 2)
    
    return res

cdef np.float64_t[:, :, :] _minCutPatch(np.float64_t[:, :, :] patch, int patchLength, int overlap, np.float64_t[:, :, :] res, int y, int x) noexcept:
    # print("overlap:", overlap)
    # print(f"x: {x}, y: {y}")  # Debugging line
    cdef int dy = patch.shape[0]
    # print("dy:", dy)
    cdef int dx = patch.shape[1]
    # print("dx:", dx)
    res = res[y : y + dy, x : x + dx]
    # print("res shape after slicing:", res.shape)
    cdef np.float64_t[:, :] leftL2
    cdef np.float64_t[:, :] upL2
    cdef int i, j
    cdef np.float64_t[:, :, :] patch_ = np.copy(patch)
    # print("patch_ shape:", patch_.shape)
    if x > 0:
        # print("x > 0")  # Debugging line
        # print("patch_[:, :overlap] shape:", patch_[:, :overlap].shape, "res[y : y + dy, x : x + overlap] shape:", res[y : y + dy, x : x + overlap].shape)  # Debugging line
        leftL2 = sum_abs_diff_axis_2(patch_[:, :overlap], res[:, :overlap])
        # print("leftL2 shape:", leftL2.shape)
        for i, j in enumerate(_minCutPath(leftL2)):
            patch_[i, :j] = res[i, :j]
            # print("minCut at row:", i, "column cut at:", j)
    if y > 0:
        # print("y > 0")  # Debugging line
        # print("patch_[:overlap, :] shape:", patch_[:overlap, :].shape, "res[y : y + overlap, x : x + dx] shape:", res[:overlap, :].shape)  # Debugging line
        upL2 = sum_abs_diff_axis_2(patch_[:overlap, :], res[:overlap, :])
        # print("upL2 shape:", upL2.shape)
        for j, i in enumerate(_minCutPath(upL2.T)):
            patch_[:i, j] = res[:i, j]
            # print("minCut at column:", j, "row cut at:", i)
    return patch_

cpdef np.ndarray[unsigned char, ndim=3] transfer(
    unsigned char [:, :, :] texture,
    unsigned char [:, :, :] target,
    int patchLength,
    str mode="cut",
    float alpha=0.1,
    int level=0,
    unsigned char [:, :, :] prior=None,
    bool blur=False,
    int overlap=0,
) noexcept:
    """
    Transfers texture from a source image (texture) onto a target image using a patch-based synthesis approach.

    This function works by breaking down the texture and target images into patches and blending them.
    The blending process can be guided by various modes, and the patches can be blended with previous results
    to achieve a multi-scale synthesis effect. The 'blur' parameter applies a Gaussian blur to the grayscale
    representation of the images to potentially reduce high-frequency artifacts during patch matching.

    Code adapted and optimized from by Alex Xu: https://github.com/axu2/image-quilting

    References:
    ----
    A. A. Efros and W. T. Freeman, “Image quilting for texture synthesis and transfer,” in Proceedings of the 28th annual conference on Computer graphics and interactive techniques, in SIGGRAPH ’01. New York, NY, USA: Association for Computing Machinery, 2001, pp. 341–346. doi: 10.1145/383259.383296.

    Args:
    - texture: The source texture image as a numpy array.
    - target: The target image where the texture will be transferred.
    - patchLength: The size of the square patch side.
    - mode: Mode of patch processing, 'cut' for minimum cut, 'best' for best correlation, or 'overlap'.
    - alpha: Weight for blending in the 'cut' mode.
    - level: The current level of the multi-scale synthesis.
    - prior: The result from the previous synthesis iteration, if available.
    - blur: Whether to apply a Gaussian blur to the images.
    - overlap: The amount of overlap between patches. If 0, defaults to 1//6 the patch length.

    Returns:
    - A numpy array representing the target image with the transferred texture.
    """
    cdef np.float64_t[:, :] corrTexture, corrTarget
    if blur:
        corrTexture = gaussian(rgb2gray(texture))
        corrTarget = gaussian(rgb2gray(target))
    else:
        corrTexture = rgb2gray(texture).astype(np.float64)
        corrTarget = rgb2gray(target).astype(np.float64)

    # remove alpha channel and convert to float
    cdef np.float64_t[:, :, :] _texture, _target
    _texture = img_as_float(texture)[:, :, :3]
    _target = img_as_float(target)[:, :, :3]

    cdef int h = _target.shape[0]
    cdef int w = _target.shape[1]
    if overlap == 0:
        overlap = max(patchLength // 6, 1)
    cdef int numPatchesHigh = int(ceil((h - overlap) / (patchLength - overlap))) + 1
    cdef int numPatchesWide = int(ceil((w - overlap) / (patchLength - overlap))) + 1
    cdef np.float64_t[:, :, :] res = _target if prior is None else img_as_float(prior)[:, :, :3]
    cdef int i, j, y, x
    cdef np.float64_t[:, :, :] patch
    
    # import cProfile
    # import pstats
    # from io import StringIO
    # pr = cProfile.Profile()
    # pr.enable()
    
    for i in range(numPatchesHigh):
        for j in range(numPatchesWide):
            y = i * (patchLength - overlap)
            x = j * (patchLength - overlap)
            if i == 0 and j == 0 or mode == "best":
                patch = _bestCorrPatch(_texture, corrTexture, patchLength, corrTarget, y, x)
            elif mode == "overlap":
                patch = _bestCorrOverlapPatch(_texture, corrTexture, patchLength, overlap, corrTarget, res, y, x)
            elif mode == "cut":
                patch = _bestCorrOverlapPatch(_texture, corrTexture, patchLength, overlap, corrTarget, res, y, x, alpha, level)
                if patch.shape[0] == 0 or patch.shape[1] == 0:
                    continue
                patch = _minCutPatch(patch, patchLength, overlap, res, y, x)
            res[y : y + patchLength, x : x + patchLength] = patch
    #
    # pr.disable()
    # s = StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    # ps.print_stats()
    # print(s.getvalue())
    
    return img_as_ubyte(res, force_copy=False)
