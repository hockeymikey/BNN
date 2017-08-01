import cupy as np


def _maybe_view_as_subclass(original_array, new_array):
  if type(original_array) is not type(new_array):
    # if input was an ndarray subclass and subclasses were OK,
    # then view the result as that subclass.
    new_array = new_array.view(type=type(original_array))
    # Since we have done something akin to a view from original_array, we
    # should let the subclass finalize (if it has it implemented, i.e., is
    # not None).
    if new_array.__array_finalize__:
      new_array.__array_finalize__(original_array)
  return new_array

class DummyArray(object):
  """Dummy object that just exists to hang __array_interface__ dictionaries
  and possibly keep alive a reference to a base array.
  """
  
  def __init__(self, interface, base=None):
    self.__array_interface__ = interface
    self.base = base

def np_lib_stride_tickets(x_padded,shape,strides):
  return np.lib.stride_tricks.as_strided(np.asarray(x_padded),shape=shape, strides=strides)
  

def as_strided(x, shape=None, strides=None, writeable=True):
    """
    Create a view into the array with the given shape and strides.

    .. warning:: This function has to be used with extreme care, see notes.

    Parameters
    ----------
    x : ndarray
        Array to create a new.
    shape : sequence of int, optional
        The shape of the new array. Defaults to ``x.shape``.
    strides : sequence of int, optional
        The strides of the new array. Defaults to ``x.strides``.
    subok : bool, optional
        .. versionadded:: 1.10

        If True, subclasses are preserved.
    writeable : bool, optional
        .. versionadded:: 1.12

        If set to False, the returned array will always be readonly.
        Otherwise it will be writable if the original array was. It
        is advisable to set this to False if possible (see Notes).

    Returns
    -------
    view : ndarray

    See also
    --------
    broadcast_to: broadcast an array to a given shape.
    reshape : reshape an array.

    Notes
    -----
    ``as_strided`` creates a view into the array given the exact strides
    and shape. This means it manipulates the internal data structure of
    ndarray and, if done incorrectly, the array elements can point to
    invalid memory and can corrupt results or crash your program.
    It is advisable to always use the original ``x.strides`` when
    calculating new strides to avoid reliance on a contiguous memory
    layout.

    Furthermore, arrays created with this function often contain self
    overlapping memory, so that two elements are identical.
    Vectorized write operations on such arrays will typically be
    unpredictable. They may even give different results for small, large,
    or transposed arrays.
    Since writing to these arrays has to be tested and done with great
    care, you may want to use ``writeable=False`` to avoid accidental write
    operations.

    For these reasons it is advisable to avoid ``as_strided`` when
    possible.
    """
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False)
    interface = dict(x.__array_interface__)
    if shape is not None:
      interface['shape'] = tuple(shape)
    if strides is not None:
      interface['strides'] = tuple(strides)
  
    array = np.asarray(DummyArray(interface, base=x))
    # The route via `__interface__` does not preserve structured
    # dtypes. Since dtype should remain unchanged, we set it explicitly.
    array.dtype = x.dtype
  
    view = _maybe_view_as_subclass(x, array)
  
    if view.flags.writeable and not writeable:
      view.flags.writeable = False
  
    return view