#
# Run Model Ensemble Experiments
#

def run(config):
    val_preds1 = config['val-preds1']
    val_preds2 = config['val-preds2']
    val_gths = config['val-gths']

    test_preds1 = config['test-preds1']
    test_preds2 = config['test-preds2']
    test_gths = config['test-gths']

    ParamManager = config['ParamManager']
    Ensembler = config['Ensembler']
    Evaluator = config['Evaluator']
    Optimizer = config['Optimizer']

    # Optimize ensemble parameters with validation set
    ensembler = Ensembler(ParamManager(), model_predictions=[val_preds1, val_preds2])
    evaluator = Evaluator(annotations=val_gths)
    optim = Optimizer(ensembler, evaluator)
    param_opt = optim.maximize()
    param_opt.dump()

    # use optimal ensemble parameters
    ev = Evaluator(annotations=test_gths)
    en = Ensembler(param_opt, model_predictions=[test_preds1, test_preds2])
    fused_preds = en.fuse()

    # as another baseline, we evaluate the fused model on default ensemble parameters
    en0 = Ensembler(ParamManager(), model_predictions=[test_preds1, test_preds2])
    fused_preds0 = en0.fuse()
    
    metric1 = ev.compute_metric(test_preds1)
    try:  # in case of hybrid ensemble
        metric2 = ev.compute_metric(test_preds2)
    except:
        ev = Evaluator(annotations=test_gths)
        metric2 = 'NA.'
    metric_fused0 = ev.compute_metric(fused_preds0)
    metric_fused  = ev.compute_metric(fused_preds)

    print("Metric Name: {}".format(str(ev)))
    print("Model1: {}".format(metric1))
    print("Model2: {}".format(metric2))
    print("Default Fused Model: {}".format(metric_fused0))
    print("Optimal Fused Model: {}".format(metric_fused))
    test_gths.dump_info()


from experiments import kangqiang_24l as generate_configuration
# from experiments import tianhe as generate_configuration
# from experiments import kangqiang_classifier_28l as generate_configuration
# from experiments import kangqiang_classifier_40l as generate_configuration
# from experiments import kangqiang_classifier_2weaks as generate_configuration
# from experiments import kangqiang_classifier_1strong as generate_configuration
# from experiments import tianhe_multilabel as generate_configuration
# from experiments import hyc_yldata as generate_configuration
# from experiments import canadian as generate_configuration
# from experiments import canadian_hybrid as generate_configuration
if __name__ == '__main__':
    run(generate_configuration())