from __future__ import print_function
import atexit, numpy as np, scipy, sys, os.path as osp
from collections import defaultdict

# ================================================================
# Math utilities
# ================================================================

def discount(x, gamma):
    """
    computes discounted sums along 0th dimension of x.

    inputs
    ------
    x: ndarray
    gamma: float

    outputs
    -------
    y: ndarray with same shape as x, satisfying

        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                where k = len(x) - t - 1

    """
    assert x.ndim >= 1
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

def explained_variance(ypred,y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def explained_variance_2d(ypred, y):
    assert y.ndim == 2 and ypred.ndim == 2
    vary = np.var(y, axis=0)
    out = 1 - np.var(y-ypred)/vary
    out[vary < 1e-10] = 0
    return out


# ================================================================
# Configuration
# ================================================================

def update_default_config(tuples, usercfg):
    """
    inputs
    ------
    tuples: a sequence of 4-tuples (name, type, defaultvalue, description)
    usercfg: dict-like object specifying overrides

    outputs
    -------
    dict2 with updated configuration
    """
    out = dict2()
    for (name,_,defval,_) in tuples:
        out[name] = defval
    if usercfg:
        for (k,v) in usercfg.iteritems():
            if k in out:
                out[k] = v
    return out

def update_argument_parser(parser, options, **kwargs):
    """
    The function updates the argument parser, by adding new options
    :param parser: Argument parser
    :param options: (list of tuples: name,type,default val, description) options to parse
    :param kwargs:
    :return:
    """
    kwargs = kwargs.copy()
    for (name,typ,default,desc) in options:
        flag = "--"+name
        if flag in parser._option_string_actions.keys(): #pylint: disable=W0212
            print("warning: already have option %s. skipping"%name)
        else:
            parser.add_argument(flag, type=typ, default=kwargs.pop(name,default), help=desc or " ")
    if kwargs:
        raise ValueError("options %s ignored"%kwargs)

def comma_sep_ints(s):
    if s:
        return map(int, s.split(","))
    else:
        return []

def IDENTITY(x):
    return x

GENERAL_OPTIONS = [
    ("seed",int,0,"random seed"),
    ("metadata",str,"","metadata about experiment"),
    ("outdir",str,"/tmp/trpo_results","output directory"),
    ("use_hdf",int,0,"whether to make an hdf5 file with results and snapshots"),
    ("snapshot_every",int,0,"how often to snapshot"),
    ("load_snapshot",str,"","path to snapshot"),
    ("video_record_every",int,1,"how often to record video. 0 == never record"),
    ("params_file",str,"config/train_params.yaml","File with training parameters")
]

# ================================================================
# Load/save
# ================================================================


def prepare_h5_file(args, out_dir):
    # Making names and opening h5 file
    if out_dir[-1] != '/':
        out_dir += '/'
    fname = out_dir + 'diagnostic.h5'

    if osp.exists(fname):
        raw_input("output file %s already exists. press enter to continue. (exit with ctrl-C)"%fname)
    import h5py
    hdf = h5py.File(fname,"w")

    # Saving parameters directly from args
    hdf.create_group('params')
    for (param,val) in args.__dict__.items():
        try: hdf['params'][param] = val
        except (ValueError,TypeError):
            print("not storing parameter",param)

    # Preparing handler for saving diagnostics that will be executed upon exit (smart move)
    diagnostics = defaultdict(list)
    print("Saving results to %s"%fname)
    def save():
        # Assumes that diagnostics is a list of values
        hdf.create_group("diagnostics")
        for (diagname, val) in diagnostics.items():
            if np.array(val[0]).ndim < 2:
                # Simple list or flattened array
                hdf["diagnostics"][diagname] = val
            else:
                # If the list of values contains np arrays we will concatenate them along first dimension
                # Thus one MUST MAKE SURE that the first dimension corresponds to iteration dimension
                # Thus if you would like to keep information per episode, make sure that you have singular dimension as the first one
                val_conc = np.concatenate(val, axis=0)
                hdf["diagnostics"][diagname] = val_conc


    # Saving command line itself
    hdf["cmd"] = " ".join(sys.argv)

    # Save will be executed upon normal exit of interpreter
    # NOTE: The functions registered via this module are not called when the program is killed by a signal not handled by Python
    atexit.register(save)

    return hdf, diagnostics


# ================================================================
# Misc
# ================================================================

class dict2(dict):
    "dictionary-like object that exposes its keys as attributes"
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return zip(*seqs)

def flatten(arrs):
    return np.concatenate([arr.flat for arr in arrs])

def unflatten(vec, shapes):
    i=0
    arrs = []
    for shape in shapes:
        size = np.prod(shape)
        arr = vec[i:i+size].reshape(shape)
        arrs.append(arr)
        i += size
    return arrs

class EzPickle(object):
    """Objects that are pickled and unpickled via their constructor
    arguments.

    Example usage:

        class Dog(Animal, EzPickle):
            def __init__(self, furcolor, tailkind="bushy"):
                Animal.__init__()
                EzPickle.__init__(furcolor, tailkind)
                ...

    When this object is unpickled, a new Dog will be constructed by passing the provided
    furcolor and tailkind into the constructor. However, philosophers are still not sure
    whether it is still the same dog.

    This is generally needed only for environments which wrap C/C++ code, such as MuJoCo
    and Atari.
    """
    def __init__(self, *args, **kwargs):
        self._ezpickle_args = args
        self._ezpickle_kwargs = kwargs
    def __getstate__(self):
        return {"_ezpickle_args" : self._ezpickle_args, "_ezpickle_kwargs": self._ezpickle_kwargs}
    def __setstate__(self, d):
        out = type(self)(*d["_ezpickle_args"], **d["_ezpickle_kwargs"])
        self.__dict__.update(out.__dict__)

def fmt_row(width, row, header=False):
    out = " | ".join(fmt_item(x, width) for x in row)
    if header: out = out + "\n" + "-"*len(out)
    return out

def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim==0
        x = x.item()
    if isinstance(x, float): rep = "%g"%x
    else: rep = str(x)
    return " "*(l - len(rep)) + rep
