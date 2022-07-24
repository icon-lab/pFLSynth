def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'pflsynth_model':
        #assert(opt.dataset_mode == 'aligned')
        from .pflsynth_model import pflsynth_model
        model = pflsynth_model()
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    #print("model [%s] was created" % (model.name()))
    return model
