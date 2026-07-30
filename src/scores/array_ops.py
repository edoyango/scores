from types import ModuleType
from typing import Union

from array_api_compat import array_namespace, is_array_api_obj

from scores.typing import is_xarraylike


def nanmean(input, dim=None, xp: Union[ModuleType, None] = None):
    """
    Dispatches to xp.nanmean or xr.DataArray/Dataset.mean(skipna=None).

    Equivalent to:
        xr:
            input.mean(dim=dim)
        xp:
            xp.nanmean(input, axis=dim)

    Notes:
        - nanmean isn't part of the strict Python Array API standard, but
          members implement it (JAX' is implemented in jax.numpy.nanmean).
    """
    if xp is not None:
        return xp.nanmean(input, axis=dim)
    elif is_xarraylike(input):
        return input.mean(dim=dim, skipna=None)
    elif is_array_api_obj(input):
        xp = array_namespace(input)
        return xp.nanmean(input, axis=dim)


def abs(input, xp: Union[ModuleType, None] = None):
    """
    Dispatches to xp.abs or np.abs.
    """
    if xp is None:
        if is_xarraylike(input):
            import numpy as np

            xp = np
        elif is_array_api_obj(input):
            xp = array_namespace(input)

    return xp.abs(input)


def where(condition, a, b, xp: Union[ModuleType, None] = None):
    """
    Dispatches to xp.where or xr.DataArray/Dataset.where.

    Equivalent to:
        xr:
            a.where(condition, b)
        xp:
            where(condition, a, b)
    """
    if xp is not None:
        return xp.where(condition, a, b)
    elif is_xarraylike(a) and is_xarraylike(b) and is_xarraylike(condition):
        return a.where(condition, b)
    elif is_array_api_obj(a) and is_array_api_obj(b) and is_array_api_obj(condition):
        xp = array_namespace(a, b, condition)
        return xp.where(condition, a, b)
