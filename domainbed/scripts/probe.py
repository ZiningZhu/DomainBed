
import argparse 
import json
from re import L
import numpy as np
import os
import random
import time
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from domainbed import algorithms 
from domainbed import datasets 
from domainbed import networks
from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.wide_resnet import Wide_ResNet
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader


class LayerwiseProbeModel(nn.Module):
    def __init__(self, algorithm, num_probe_classes):
        super(LayerwiseProbeModel, self).__init__()
        self.algorithm = algorithm 
        self.num_probe_classes = num_probe_classes
        self.probe_init_lr = 1e-4
        
        self.probing_classifiers = nn.ModuleList([])
        self.optimizers = []
        if hasattr(algorithm, "featurizer"):
            if isinstance(algorithm.featurizer, networks.MLP):
                for hid in algorithm.featurizer.hiddens:
                    probing_clf = nn.Linear(hid.out_features, self.num_probe_classes)
                    self.probing_classifiers.append(probing_clf)
                    self.optimizers.append(torch.optim.Adam(probing_clf.parameters(), lr=self.probe_init_lr))
            elif isinstance(algorithm.featurizer, networks.MNIST_CNN):
                hid_dims = [50176, 25088, 25088, 25088, 128]
                for i in range(len(hid_dims)):
                    probing_clf = nn.Linear(hid_dims[i], num_probe_classes)
                    self.probing_classifiers.append(probing_clf)
                    self.optimizers.append(torch.optim.Adam(probing_clf.parameters(), lr=self.probe_init_lr))

            elif isinstance(algorithm.featurizer, networks.ResNet):
                raise NotImplementedError("TODO")
            elif isinstance(algorithm.featurizer, Wide_ResNet):
                raise NotImplementedError("TODO")
            else:
                raise ValueError("Probing of the featurizer (class {}) of algorithm {} is not supported yet!".format(algorithm.featurizer.__class__, algorithm.__class__))
        else:
            raise ValueError("Algorithm {} does not have a featurizer to probe.".format(algorithm.__class__))
        
    def update(self, minibatches, unlabeled=None):
        # Train all probes for one step
        # TODO - what if different probes need different steps for training?
        # minibatches contain batches from different environments (as labeled by y)
        all_x, all_y = [], []
        for x,y in minibatches:
            all_x.append(x)
            all_y.append(y)
        all_x = torch.cat(all_x)
        all_y = torch.cat(all_y)

        logits, batch_representations = self.algorithm.featurizer.probe_forward(all_x)
        for pid, rep in enumerate(batch_representations):
            loss = F.cross_entropy(self.probing_classifiers[pid](rep), all_y)
            loss.backward()
            self.optimizers[pid].step()
            self.optimizers[pid].zero_grad()

    def predict(self, x):
        logits, batch_representations = self.algorithm.featurizer.probe_forward(x)
        preds = []
        for pid, rep in enumerate(batch_representations):
            pred = self.probing_classifiers[pid](rep)  # (bsz, dim_out)
            maxval, maxid = torch.max(pred, dim=-1)
            preds.append(maxid)
        return preds 

def evaluate_probe_accuracy(probe, 
                eval_loader_env_labels, 
                eval_loaders,
                eval_weights, device, suffix):
    """
    Return: a dictionary (name -> scalar)
    """ 
    
    evals = zip(eval_loader_env_labels, 
        eval_loaders, eval_weights)
    correct, total = {}, {}  # Dict of (N. probes * N. env) entries
    for envid, loader, weight in evals:
        # loader contains (x,y) but we use our env_label as y instead
        for x, _ in loader:
            x = x.to(device)
            y = envid * torch.ones(len(x)).to(device)
            
            preds = probe.predict(x)
            for pid, pred in enumerate(preds):
                nc = (y==pred).sum().item()
                nt = len(x)
                k = (pid, envid)
                if k not in correct:
                    correct[k] = nc
                    total[k] = nt
                else:
                    correct[k] += nc 
                    total[k] += nt  
    results = {}
    for k in correct:
        pid, envid = k
        results[f"probe_{pid}_env_{envid}_{suffix}"] = correct[k] / total[k]
    return results 


def main(probe_args):
    checkpoint = torch.load(
        os.path.join(probe_args.checkpoint_dir, "model.pkl"))
    train_args = argparse.Namespace(**checkpoint["args"])
    algorithm_dict = checkpoint["model_dict"]

    if train_args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(train_args.algorithm, train_args.dataset)
    else:
        hparams = hparams_registry.random_hparams(train_args.algorithm, train_args.dataset,
            misc.seed_hash(train_args.hparams_seed, train_args.trial_seed))
    if train_args.hparams:
        hparams.update(json.loads(train_args.hparams))

    random.seed(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if train_args.dataset in vars(datasets):
        dataset = vars(datasets)[train_args.dataset](train_args.data_dir,
            train_args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We train the probes on all in-split (except the test env), and evaluate on out-splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*train_args.holdout_fraction),
            misc.seed_hash(train_args.trial_seed, env_i))

        if env_i in train_args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*train_args.uda_holdout_fraction),
                misc.seed_hash(train_args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if train_args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    algorithm_class = algorithms.get_algorithm_class(train_args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, len(dataset) - len(train_args.test_envs), hparams)
    algorithm.load_state_dict(algorithm_dict)

    probe = LayerwiseProbeModel(algorithm, len(dataset) - len(train_args.test_envs))
    probe.to(device)

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in train_args.test_envs]
    eval_loaders_insplits = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in in_splits]
    eval_weights_insplits = [None for _, weights in in_splits]
    eval_loader_env_labels = [i for i in range(len(in_splits))]

    eval_loaders_outsplits = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in out_splits]
    eval_weights_outsplits = [None for _, weights in in_splits]

    train_minibatches_iterator = zip(*train_loaders)
    
    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])
    n_steps = probe_args.steps or dataset.N_STEPS 
    start_time = time.time()

    for step in range(0, n_steps):
        minibatches_device = []
        for envid, (x,y) in enumerate(next(train_minibatches_iterator)):
            envlabels = envid * torch.ones_like(y)
            minibatches_device.append((x.to(device), envlabels.to(device)))
        step_vals = probe.update(minibatches_device)

        if (step % probe_args.report_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
                'time_per_epoch': (time.time() - start_time) / (step / steps_per_epoch)
            }
            
            probe_train_accs = evaluate_probe_accuracy(probe, 
                eval_loader_env_labels, 
                eval_loaders_insplits,
                eval_weights_insplits, device, "insplit")
            probe_test_accs = evaluate_probe_accuracy(probe, 
                eval_loader_env_labels, 
                eval_loaders_outsplits,
                eval_weights_outsplits, device, "outsplit")
            results = {**results, **probe_train_accs, **probe_test_accs}
            
            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            report_path = os.path.join(probe_args.report_dir, 'probe_results_{}.jsonl'.format(probe_args.slurm_id))
            with open(report_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")
    print("Probing done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Probe')
    parser.add_argument('--checkpoint_dir', type=str, default="train_output")
    parser.add_argument('--steps', type=int, default=0, help='If 0, use the total steps necessary to traverse the dataset')
    parser.add_argument('--report_freq', type=int, default=100)
    parser.add_argument('--report_dir', type=str, default="", help='If empty, use the same directory of the checkpoint')
    parser.add_argument("--slurm_id", type=str, default=0)
    probe_args = parser.parse_args()

    if not probe_args.report_dir:
        probe_args.report_dir = probe_args.checkpoint_dir 

    print(probe_args)

    main(probe_args)