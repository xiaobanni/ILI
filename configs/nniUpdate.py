def nni_update(cfg, optimized_params):
    """Update the config with the optimized parameters from NNI.

    Args:
        cfg (UtilsRL.misc.namespace.NameSpaceMeta): defualt config.
        optimized_params (dict): optimized parameters from NNI.
    """
    for key, value in cfg._data_.items():
        if isinstance(value, type):
            nni_update(value, optimized_params)
        else:
            if key in optimized_params:
                setattr(cfg, key, optimized_params[key])
    return cfg
