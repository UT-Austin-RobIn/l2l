import collections
import h5py
import numpy as np

class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d  

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def prefix_dict(d, prefix):
    """Adds the prefix to all keys of dict d."""
    return type(d)({prefix + '/' + k: v for k, v in d.items()})


def recursive_map_dict(fn, d, ignore=[]):
    '''
        fn: function to apply to each element
        d: a dictionary of lists
        ignore: list of keys not to convert to numpy
    '''

    new_d = {}
    for k in d.keys():
        if k in ignore:
            new_d[k] = d[k]
        elif isinstance(d[k], dict) or isinstance(d[k], h5py._hl.group.Group):
            new_d[k] = recursive_map_dict(fn, d[k], ignore=ignore)
        else:
            new_d[k] = fn(d[k])

    return new_d

def listdict2dictlist(ld):
    '''
        ld: list of dictionary
    '''
    
    def recursive_dict_merge(a, b):
        '''
            appending b to a
            a may not have the keys of b
        '''
        for k in b:
            if isinstance(b[k], dict):
                a[k] = recursive_dict_merge(a[k] if k in a else {}, b[k])
            else:
                if k in a:
                    a[k].append(b[k])
                else:
                    a[k] = [b[k]]
                # a[k] = a[k] + b[k] if k in a else b[k] # this is an append operation
        # print(a)
        return a
    
    dl = {}
    for d in ld:
        dl = recursive_dict_merge(dl, d)

    return dl


def flatten_dict(dct):
    
    def recursively_flatten(d, new_d):
        for k in d:
            if isinstance(d[k], dict):
                recursively_flatten(d[k], new_d)
            else:
                new_d[k] = d[k]
        return new_d
    
    return recursively_flatten(dct, {})
    

def get_env_from_config(config, seed=None):
    try:
        import robosuite as suite
        env = suite.make(**config.env_config)
    except:
        env = config.env_config.env_class(**config.env_config.env_kwargs)
    for wrapper in config.wrappers:
        env = wrapper(env)

    if seed is not None:
        env.reset(seed=seed)
    return env

def recursive_dict_write_in_grp(grp, data):
    for k in data.keys():
        if isinstance(data[k], dict):
            sub_grp = grp.create_group(k)
            recursive_dict_write_in_grp(sub_grp, data[k])
        else:
            grp.create_dataset(k, data=data[k])


if __name__=='__main__':
    a = [{'a': 2, 'b':{'bb': 5}}, {'a': 34, 'b':{'bb': 7}}, {'a': 24, 'b':{'bb': 10}}]

    print(listdict2dictlist(a))