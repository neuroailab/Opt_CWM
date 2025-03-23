import os
from pprint import pprint

import yaml
from easydict import EasyDict as edict  # pip install easydict

from utils import dist_logging

logger = dist_logging.get_logger(__name__)


def parse_arguments(args):
    # parse from command line (syntax: --key1.key2.key3=value)
    opt_cmd = {}
    for arg in args:
        assert arg.startswith("--")
        if "=" not in arg[2:]:  # --key means key=True, --key! means key=False
            key_str, value = (arg[2:-1], "false") if arg[-1] == "!" else (arg[2:], "true")
        else:
            key_str, value = arg[2:].split("=")
        keys_sub = key_str.split(".")
        opt_sub = opt_cmd
        for k in keys_sub[:-1]:
            if k not in opt_sub:
                opt_sub[k] = {}
            opt_sub = opt_sub[k]
        # if opt_cmd['key1']['key2']['key3'] already exist for key1.key2.key3, print key3 as error msg
        assert keys_sub[-1] not in opt_sub, keys_sub[-1]
        logger.info(f"Setting config field from CLI: {key_str}")
        opt_sub[keys_sub[-1]] = yaml.safe_load(value)
    opt_cmd = edict(opt_cmd)
    return opt_cmd


def set(opt_cmd={}, verbose=True, safe_check=True):
    fname = opt_cmd.yaml  # load from yaml file
    opt_base = load_options(fname)
    # override with command line arguments
    opt = override_options(opt_base, opt_cmd, key_stack=[], safe_check=safe_check)
    if verbose:
        pprint(opt)
    return opt


# this recursion seems to only work for the outer loop when dict_type is not dict
def to_dict(D, dict_type=dict):
    D = dict_type(D)
    for k, v in D.items():
        if isinstance(v, dict):
            D[k] = to_dict(v, dict_type)
    return D


def load_options(fname):
    def tuple_constructor(loader, node):
        return tuple(loader.construct_sequence(node))

    # Add the constructor to the YAML loader
    yaml.add_constructor("tag:yaml.org,2002:python/tuple", tuple_constructor)

    with open(fname) as file:
        # opt = edict(yaml.safe_load(file))
        opt = edict(yaml.load(file, yaml.Loader))
    if "_parent_" in opt:
        # load parent yaml file(s) as base options
        parent_fnames = opt.pop("_parent_")
        if type(parent_fnames) is str:
            parent_fnames = [parent_fnames]
        for parent_fname in parent_fnames:
            opt_parent = load_options(parent_fname)
            opt_parent = override_options(opt_parent, opt, key_stack=[])
            opt = opt_parent
    logger.info("loading {}...".format(fname))
    return opt


def override_options(opt, opt_over, key_stack=None, safe_check=False):
    for key, value in opt_over.items():
        if isinstance(value, dict):
            # parse child options (until leaf nodes are reached)
            opt[key] = override_options(opt.get(key, dict()), value, key_stack=key_stack + [key], safe_check=safe_check)
        else:
            # ensure command line argument to override is also in yaml file
            if safe_check and key not in opt and key != "yaml":
                add_new = None
                while add_new not in ["y", "n"]:
                    key_str = ".".join(key_stack + [key])
                    add_new = input('"{}" not found in original opt, add? (y/n) '.format(key_str))
                if add_new == "n":
                    print("safe exiting...")
                    exit()
            opt[key] = value
    return opt


def save_options_file(path, opt):
    opt_fname = os.path.join(path, "config.yaml")
    with open(opt_fname, "w") as file:
        yaml.safe_dump(to_dict(opt), file, default_flow_style=False, indent=4)
