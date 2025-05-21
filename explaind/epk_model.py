import torch
from tqdm import tqdm

from explaind.modulo_model.loss import RegularizedCrossEntropyLoss
from explaind.model_paths import ModelPath
from explaind.optimizer_paths import OptimizerPath
from explaind.data_paths import DataPath


device = "cuda:0" if torch.cuda.is_available() else "cpu"

class ExactPathKernelModel:
    def __init__(self, 
                 model : ModelPath,
                 optimizer : OptimizerPath,
                 loss_fn : torch.nn.Module,
                 # train_loader,
                 data_path : DataPath,
                 integral_eps=0.1,
                 train_set_size=None,
                 features_to_cpu=False,
                 device=device,
                 evaluate_predictions=True,
                 early_integral_eps=0.001,
                 inference_batch_size=99999999,
                 grad_scaling=1.0,
                 kernel_store_interval=100,
                 keep_param_wise_kernel=False,
                 param_wise_kernel_keep_out_dims=False,
                 early_steps=4):
        
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = data_path.dataloader
        self.integral_eps = integral_eps
        self.device = device
        self.features_to_cpu = features_to_cpu
        self.evaluate_predictions_flag = evaluate_predictions
        self.prediction_deltas = []
        self.early_integral_eps = early_integral_eps
        self.early_steps = early_steps
        self.train_set_size = data_path.dataloader.dataset.__len__() if train_set_size is None else train_set_size
        self.data_path = data_path
        self.inference_batch_size = inference_batch_size
        self.grad_scaling = grad_scaling
        self.keep_param_wise_kernel = keep_param_wise_kernel
        self.param_wise_kernel_keep_out_dims = param_wise_kernel_keep_out_dims
        self.kernel_store_interval = kernel_store_interval

    
    def predict(self, X_test, y_test=None, is_train=False, keep_kernel_matrices=False):

        steps = self.get_integral_steps(self.integral_eps, pred_is_train=is_train, 
                                        early_eps=self.early_integral_eps, early_steps=self.early_steps)

        initial_prediction = self.model.forward(X_test, step=0)

        print("Initial prediction:", initial_prediction, initial_prediction.shape)

        current_prediction = initial_prediction.clone()
        if self.features_to_cpu:
            current_prediction = current_prediction.cpu()
            initial_prediction = initial_prediction.cpu()

        next_reg = None
        prev_eval = None
        step_reg_features = None

        # todo: generalize to other models
        full_pred_target = torch.diag(torch.ones(initial_prediction.shape[1], device=self.device, dtype=torch.float32)).to(self.device)

        # init next_avg
        next_avg = None

        overall_pred_kernel = None
        overall_reg_term = None

        for i, step in tqdm(enumerate(steps)):

            train_step, pred_steps = step

            batch = self.data_path.get_checkpoint(train_step)
            ids = batch["indices"]
            X_train, y_train = batch["X"].to(self.device), batch["y"].to(self.device)

            
          
            if is_train:
                # this is not correctly implemented yet
                step_kernel, next_avg, data_features = self.exact_path_kernel_step_train(X_test, 
                                                                                         X_train,
                                                                                         y_train,
                                                                                         next_avg, 
                                                                                         train_step, 
                                                                                         pred_target=full_pred_target)
            else:
                step_kernel, next_avg, data_features, prev_eval = self.exact_path_kernel_step_test(X_test, 
                                                                                        X_train,
                                                                                        y_train,
                                                                                        next_avg, 
                                                                                        train_step, 
                                                                                        pred_target=full_pred_target,
                                                                                        pred_steps=pred_steps,
                                                                                        ids=ids,
                                                                                        y_test=y_test,
                                                                                        prev_eval=prev_eval)


            step_prediction = self.get_prediction_from_kernel(step_kernel, batch_size=len(ids))

            lr = self.optimizer.get_learning_rate(train_step + 1)

            current_prediction = current_prediction - lr * step_prediction

            step_reg_term, step_reg_features = self.optimizer.get_reg_term(data_features, 
                                                                           step_reg_features, 
                                                                           train_step,
                                                                           prev_eval=prev_eval,
                                                                           param_kernel_store_interval=self.kernel_store_interval,)
            current_prediction = current_prediction + step_reg_term   
            
            if self.evaluate_predictions_flag:
                prev_eval = self.evaluate_predictions(X_test, 
                                          current_prediction, 
                                          train_step, 
                                          -step_prediction * lr,
                                          prev_eval=prev_eval,
                                          reg_term=step_reg_term)

            if keep_kernel_matrices:
                overall_pred_kernel, prev_eval = self.update_kernel_matrix(overall_pred_kernel, step_kernel, ids, lr, step=train_step, prev_eval=prev_eval)
                overall_reg_term = self.update_regularization_term(overall_reg_term, step_reg_term, lr)
            
            # torch.cuda.empty_cache()
            
        return current_prediction, initial_prediction, overall_pred_kernel, overall_reg_term, prev_eval
    
    def update_kernel_matrix(self, kernel, step_kernel, ids, lr, step=0, prev_eval=None):
    
        if kernel is None:
            kernel = {name: torch.zeros((step_kernel[name].shape[0], step_kernel[name].shape[1], 
                                         step_kernel[name].shape[2], len(self.data_path.dataloader.dataset)), device=self.device) for name in step_kernel.keys()}

        for k, v in step_kernel.items():
            if v.shape[-1] == len(ids):
                kernel[k][:, :, :, ids] = kernel[k][:, :, :, ids] + v * lr
            else:
                kernel[k] = kernel[k] + v * lr

        if prev_eval is not None:
            if  step % self.kernel_store_interval == 0:
                if "kernel_influence_val" in prev_eval:
                    prev_eval["kernel_step_matrix"].append({k: kernel[k].sum(dim=1).sum(dim=1).detach().clone().cpu() for k in kernel.keys()})
                else:
                    prev_eval["kernel_step_matrix"] = [{k: kernel[k].sum(dim=1).sum(dim=1).detach().clone().cpu() for k in kernel.keys()}]

        return kernel, prev_eval
    
    def update_regularization_term(self, reg_term, step_reg, lr):
        if reg_term is None:
            reg_term = torch.zeros(step_reg.shape, device=self.device)
        
        reg_term = reg_term + step_reg * lr

        return reg_term
    
    def get_prediction_from_kernel(self, kernel, batch_size=4000):
        # sum up parameter-wise kernel vals
        combined = None
        for k, v in kernel.items():
            # sum up over train data dim
            v_summed = torch.sum(v, dim=-1).reshape(v.shape[0], v.shape[2])
            if combined is None:
                combined = v_summed
            else:
                combined = combined + v_summed
        return combined * (-1.0) / batch_size
    

    def evaluate_predictions(self, X_test, other_prediction, train_step, step, prev_eval, reg_term):

        if len(X_test.shape) == 1:
            X_test = X_test.unsqueeze(0)

        other_prediction = other_prediction.clone().detach().cpu()
        step = step.clone().detach().cpu()
        reg_term = reg_term.clone().detach().cpu()
        current_output = self.model.forward(X_test.to(self.device), step=train_step+1).clone().detach().cpu()
        prev_output = self.model.forward(X_test.to(self.device), step=train_step).clone().detach().cpu()

        ground_truth_step = current_output - prev_output

        evaluation = {}
        evaluation["ground_truth_step"] = ground_truth_step
        evaluation["predicted_step"] = step + reg_term
        evaluation["regularization_term"] = reg_term

        evaluation["step_l1_norm"] = torch.sum(torch.abs(step)).item()
        evaluation["step_l2_norm"] = torch.sqrt(torch.sum(step ** 2)).item()
        evaluation["regularization_term_l1_norm"] = torch.sum(torch.abs(reg_term)).item()
        evaluation["regularization_term_l2_norm"] = torch.sqrt(torch.sum(reg_term ** 2)).item()

        evaluation["predictions_mean_delta"] = torch.sum(torch.abs(current_output - other_prediction)).item() / current_output.shape[1] / current_output.shape[0]
        evaluation["predictions_max_delta"] = torch.max(torch.abs(current_output - other_prediction)).item()

        evaluation["step_mean_delta"] = torch.sum(torch.abs(ground_truth_step - step - reg_term)).item() / ground_truth_step.shape[1] / ground_truth_step.shape[0]
        evaluation["step_max_delta"] = torch.max(torch.abs(ground_truth_step - step - reg_term)).item()

        print(f"Prediction-changes differ at step {train_step} by {evaluation['step_mean_delta']} on avg. (Max diff: {evaluation['step_max_delta']})")
        print(f"Predictions differ at step {train_step} by {evaluation['predictions_mean_delta']} on avg. (Max diff: {evaluation['predictions_max_delta']})")
        
        if prev_eval is None:
            prev_eval = {k: [v] for k, v in evaluation.items()}
        else:
            for k, v in evaluation.items():
                if k not in prev_eval.keys():
                    prev_eval[k] = [v]
                else:
                    prev_eval[k].append(v)
        
        return prev_eval



    def get_loss_grad(self, X, y, step):
        # todo for other loss types
        
        loss_grads = []
        if type(self.loss_fn) is RegularizedCrossEntropyLoss or type(self.loss_fn) is torch.nn.CrossEntropyLoss:
            target_mask = self.get_target_mask(y, device=self.device)
            loss_grads = target_mask * -1
        else:
            print(f"Loss type {type(self.loss_fn)} has no simplified/efficient loss gradient computation implemented. Falling back to full Jacobian computation (which can be very slow).")
            print("If you're surprised by this message and think there should be a more efficient way to compute the loss gradient, please open an issue on the path_kernels GitHub repository.")

            print("Full Jacobian not implemented yet.")
            
            output = self.model(X.to(self.device)).detach()
            loss = self.loss_fn(output, y.to(self.device))
            grads = torch.autograd.grad(loss, output, create_graph=True)
            loss_grads = grads             

        loss_grads = loss_grads.reshape(loss_grads.shape[0], -1)

        return loss_grads


    def exact_path_kernel_step_test(self, 
                                    X_test,
                                    X_train,
                                    y_train,
                                    last_avg,
                                    train_step, 
                                    pred_target,
                                    pred_steps,
                                    ids,
                                    y_test=None,
                                    prev_eval=None): 

        # these are only computed for the correct target y (rest is zero for CE loss)
        # todo: base this decision on the loss function
        if type(self.loss_fn) is RegularizedCrossEntropyLoss or type(self.loss_fn) is torch.nn.CrossEntropyLoss:
            train_features = self.model.step_gradient_feature_map(X_train, y_train, train_step)
        else:
            raise NotImplementedError("Only CrossEntropyLoss is supported for now.")
        
        train_features, next_avg = self.optimizer.train_feature_map(train_features, 
                                                                    last_avg, 
                                                                    train_step, 
                                                                    ids=ids, 
                                                                    dataset_size=self.train_set_size)
        
        data_features = self.model.gradient_step_integral(X_test, 
                                                          None,
                                                          train_step, 
                                                          train_step + 1, 
                                                          pred_target, 
                                                          pred_steps,
                                                          optimizer=self.optimizer)

        if self.keep_param_wise_kernel:
            param_step_kernel = self.optimizer.get_param_wise_kernel(data_features, train_features, to_cpu=False, 
                                                                     keep_out_dims=self.param_wise_kernel_keep_out_dims)
            # concat different mocel parts
            for k, v in param_step_kernel.items():
                param_step_kernel[k] *= self.optimizer.get_learning_rate(train_step + 1) * (1 / len(y_train))

            if prev_eval is None:
                prev_eval = {}
            if "param_kernel" not in prev_eval.keys():
                prev_eval["param_kernel"] = param_step_kernel
            else:
                for k, v in param_step_kernel.items():
                    prev_eval["param_kernel"][k] += v

            if train_step % self.kernel_store_interval == 0 and y_test is not None:
                step_key = f"param_kernel_step_{train_step}"
                # get the slice of the positive class
                param_kernel_acc = {k: v.sum(dim=1).sum(dim=1).detach().clone().cpu() for k, v in prev_eval["param_kernel"].items()}

                prev_eval[step_key] = param_kernel_acc
                # reset
                prev_eval["param_kernel"] = {k: torch.zeros(v.shape, device=self.device) for k, v in param_step_kernel.items()}


        step_kernel, next_avg = self.optimizer.get_step_kernel(data_features, train_features, train_step, next_avg, ids)


        # store step_kernel
        store_step_kernel = {}
        if train_step % self.kernel_store_interval == 0:
            for k, v in step_kernel.items():
                store_step_kernel[k] = v.sum(dim=1).sum(dim=1).detach().cpu().clone()

            prev_eval[f"kernel_step_matrix_{train_step}"] = store_step_kernel

        last_avg = next_avg

        return step_kernel, next_avg, data_features, prev_eval

    
    
    def exact_path_kernel_step_train(self, X_test, X_train, y_train, last_avg, train_step, pred_target): 
        
        # todo: see above todo and perhaps combine functions for better code reuse
        train_features = self.model.step_gradient_feature_map(X_train, y_train, train_step)
        train_features, next_avg = self.optimizer.train_feature_map(train_features, last_avg, train_step)
        
        data_features = self.model.step_gradient_feature_map(X_test, None,
                                                            train_step, 
                                                            pred_target)

        # step_kernel = {name: torch.matmul(data_features[name], train_features[name].T) for name in data_features.keys()}
        step_kernel = {name: torch.einsum("abc,dec->adbe", data_features[name], train_features[name]) for name in data_features.keys()}
    
        return step_kernel, next_avg, data_features
    
    def get_integral_steps(self, eps, pred_is_train=False, early_eps=None, early_steps=4):

        train_steps = []
        pred_steps = []

        if early_eps is not None:
            early_eps = min(early_eps, eps)
            param_step_size = max(early_eps, 1)

            integral_no_steps = int(max(1, 1 / early_eps))
            integral_step_size = 1 / integral_no_steps

            if integral_step_size > 0.5:
                integral_step_size = 1

            for i in range(0, early_steps, param_step_size):
                train_steps.append(i)
                pred_intervals = []

                if pred_is_train:
                    pred_intervals.append([0])
                else: 
                    for j in range(0, integral_no_steps):
                        pred_intervals.append(j * integral_step_size + integral_step_size/2)
                pred_steps.append(pred_intervals)
        
        param_step_size = max(eps, 1)
        integral_no_steps = int(max(1, 1 / eps))
        integral_step_size = 1 / integral_no_steps

        if integral_step_size > 0.5:
            integral_step_size = 1

        for i in range(early_steps, len(self.model.checkpoints), param_step_size):
            train_steps.append(i)
            pred_intervals = []

            if pred_is_train:
                pred_intervals.append([0])
            else: 
                for j in range(0, integral_no_steps):
                    pred_intervals.append(j * integral_step_size + integral_step_size/2)  # mid point rule

            pred_steps.append(pred_intervals)

        # last step is the final model
        return list(zip(train_steps, pred_steps))[:-1]