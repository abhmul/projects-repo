def experiment_logfile(number):
    return f"experiment_results/experiment{number}_results.txt"

def model_extra_summary(model):
    def with_config(obj):
        return {"obj_name": type(obj).__name__, **obj.get_config()}

    print(f"Model layer kernels initialized with {[with_config(layer.kernel_initializer) for layer in model.layers]}")
    print(f"Model layer biases initialized with {[with_config(layer.bias_initializer) for layer in model.layers]}")
    print(f"Model training with: optimizer {with_config(model.optimizer)}")
    print(f"Model training with loss {with_config(model.compiled_loss._losses)}")
    print(f"Model training with metrics {[with_config(metric) for metric in model.compiled_metrics.metrics]}")
