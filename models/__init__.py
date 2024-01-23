def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'federated_synthesis':
        from .federated_synthesis import federated_synthesis as federated_synthesis
        model = federated_synthesis()
    elif opt.model == 'fedmm':
        from .fedmmgan_model import fedmmgan_model
        model = fedmmgan_model()
    elif opt.model == 'switchable_cycle_gan':
        from .switchable_cycle_gan_model import SwitchableCycleGANModel
        model = SwitchableCycleGANModel()
    elif opt.model == 'centralized_model':
        from .centralized_model import centralized_model
        model = centralized_model()
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
