import os

def overwrite_params( args, config ):

    # overwrite some config parameters given by commandline arguments
    if( args.myarg_batch_size != -1 ):
        if( 'train' in config.dataset ):
            config.dataset.train.others.bs = args.myarg_batch_size
        if( 'val' in config.dataset ):
            config.dataset.val.others.bs = args.myarg_batch_size
        if( 'test' in config.dataset ):
            config.dataset.test.others.bs = args.myarg_batch_size
        if( 'train_source' in config.dataset ):
            config.dataset.train_source.others.bs = args.myarg_batch_size
        if( 'train_target' in config.dataset ):
            config.dataset.train_target.others.bs = args.myarg_batch_size
    if( args.myarg_trans_depth != -1 ):
        config.model.depth = args.myarg_trans_depth
        if( 'transformer_config' in config.model ):
            config.model.transformer_config.depth = args.myarg_trans_depth
    if( args.myarg_group_size != -1 ):
        config.model.group_size = args.myarg_group_size
    if( args.myarg_num_group != -1 ):
        config.model.num_group = args.myarg_num_group
    if( args.myarg_mask_ratio != -1 ):
        config.model.mask_ratio = args.myarg_mask_ratio
        if( 'transformer_config' in config.model ):
            config.model.transformer_config.mask_ratio = args.myarg_mask_ratio

    # newly add some config parameters, which are given by commandline arguments
    config.rot_train = args.myarg_rot_train
    config.rot_test = args.myarg_rot_test
    config.model.lrf_axis1 = args.myarg_lrf_axis1
    config.model.lrf_axis2 = args.myarg_lrf_axis2
    config.model.output_feature_type = args.myarg_output_feature_type
    config.model.pod_gridsize = args.myarg_pod_gridsize
    config.model.attn_mode = args.myarg_attn_mode
    config.model.attn_subsample_rate = args.myarg_attn_subsample_rate
    config.model.attn_target = args.myarg_attn_target

    return args, config
