import enum
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch
from torch.utils.data import DataLoader

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def subset(dataloader, subset_fraction=0.5):
    total_size = len(dataloader.dataset)
    indices = torch.randperm(total_size)[:int(subset_fraction * total_size)]
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    return DataLoader(dataloader.dataset, batch_size=dataloader.batch_size, sampler=sampler)

class FileLogger():
    def __init__(self, file_path):
        self.file = open(file_path, 'a', buffering=1)

    def print(self, *args, **kwargs):
        print(*args, file=self.file, **kwargs)

class ResumeSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, start_idx=0, shuffle=False, seed=0, epoch=0):
        if shuffle:
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(seed + epoch)
            self.indices = torch.randperm(len(data_source))[start_idx:].tolist()
            torch.random.set_rng_state(rng_state)
        else:
            self.indices = torch.arange(len(data_source))[start_idx:].tolist()

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class Metric():
    def __init__(self, abbrev, comp_f, default_val, format_str, suffix=None):
        self.abbrev = abbrev
        self.comp_f = comp_f
        self.default_val = default_val
        self.format_str = format_str
        self.suffix = suffix

    def compute(self, model, data, losses):
        return self.comp_f(model, data, losses)

    def format(self, val):
        return '{val:{format}}'.format(val=val, format=self.format_str) + (self.suffix or '')

    def default_str(self):
        return ' ' * len(self.format(self.default_val))

class ResumableModel(torch.nn.Module):
    def __init__(self, hyperparameters, training_settings, base_dir, model: torch.nn.Module = None, criterion = None, optimizer: torch.optim.Optimizer = None, custom_metrics: list[Metric] = None, seed=0):
        super(ResumableModel, self).__init__()
        self.hyperparameters = hyperparameters
        self.training_settings = training_settings
        self.base_dir = base_dir
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.custom_metrics = custom_metrics

        self.lr = self.hyperparameters['lr']
        self.num_epochs = self.training_settings['num_epochs']
        self.max_patience = self.training_settings['max_patience']
        self.print_interval = self.training_settings['print_interval']
        self.eval_subset_interval = self.training_settings['eval_subset_interval']
        self.eval_full_interval = self.training_settings['eval_full_interval']
        self.eval_subset_fraction = self.training_settings['eval_subset_fraction']

        name = self.get_name()
        log_path = f'{base_dir}/logs/{name}.log'
        self.logger = FileLogger(log_path)

        self.seed = seed

        self.initialize()

    def reset_patience(self):
        self.patience = self.max_patience

    def initialize(self):
        self.start_epoch = 0
        self.start_batch = 0
        self.reset_patience()

        self.eval_indices = [0]
        self.train_loss = [float('inf')]
        self.val_loss = [float('inf')]
        self.test_loss = [float('inf')]
        self.best_loss = float('inf')

    def get_name(self):
        name = ''
        for arg, val in self.hyperparameters.items():
            name += f'{arg}_{val}_'
        name = name[:-1]
        return name

    def get_model_path(self, suffix):
        return f'{self.base_dir}/models/{self.get_name()}_{suffix}.pth'

    def get_num_params(self):
        return sum(p.numel() for p in self.model.parameters())/1e6

    def set_model(self, model):
        self.model = model

    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_mode(self, train=False, eval=False):
        if train:
            self.model.train()
        elif eval:
            self.model.eval()
        else:
            raise TypeError('`train` or `eval` must be set to True')

    def forward(self, x):
        return self.model(x)

    def try_load(self, suffix=None):
        try:
            if suffix is None:
                suffix = input('Suffix?')
            checkpoint = torch.load(self.get_model_path(suffix))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            rng_state = checkpoint['rng_state']
            self.seed = rng_state['seed']
            torch.set_rng_state(rng_state['torch'])
            torch.cuda.set_rng_state_all(rng_state['cuda'])
            random.setstate(rng_state['random'])
            np.random.set_state(rng_state['numpy'])

            self.start_epoch = checkpoint['epoch']
            self.epoch = self.start_epoch
            self.start_batch = checkpoint['batch']
            self.batch = self.start_batch

            self.eval_indices = checkpoint['eval_indices']
            self.train_loss = checkpoint['train_loss']
            self.val_loss = checkpoint['val_loss']
            self.test_loss = checkpoint['test_loss']
            self.best_loss = self.val_loss[-1]

            self.logger.print(f'Starting from epoch {self.start_epoch}, batch {self.start_batch}, loss {self.best_loss:.4f}')

        except FileNotFoundError:
            self.initialize()
            self.logger.print('Starting from scratch (No checkpoint found)')

    def save(self, suffix):
        torch.save({
            'epoch': self.epoch,
            'batch': self.batch + 1,
            'eval_indices': self.eval_indices,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'test_loss': self.test_loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'rng_state': {
                'seed': self.seed,
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all(),
                'random': random.getstate(),
                'numpy': np.random.get_state(),
            }
        }, self.get_model_path(suffix))

    def train(self, data):
        self.time_start = time.time()
        self.t_interval_end = time.time()

        if self.start_batch == len(data['train']):
            self.start_epoch += 1
            self.start_batch = 0

        try:
            self.do_train(data)
        except KeyboardInterrupt:
            self.logger.print('Stopping, user requested')

        self.logger.print(f'Trained to E {self.epoch}, B {self.batch} for {(time.time() - self.time_start):4.1f}s')

        self.save(f'final_E{self.epoch}_B{self.batch}')

    def do_train(self, all_data):
        num_batches = len(all_data['train'])

        for epoch in range(self.start_epoch, self.num_epochs):
            self.epoch = epoch

            if epoch != self.start_epoch:
                self.start_batch = 0

            data_tr = all_data['train']
            sampler = ResumeSampler(data_tr.dataset, self.start_batch * data_tr.batch_size, shuffle=True, seed=self.seed, epoch=epoch)
            data_tr = DataLoader(data_tr.dataset, batch_size=data_tr.batch_size, sampler=sampler)

            for batch, (data, targets) in enumerate(data_tr, start=self.start_batch):
                self.batch = batch
                self.iter = epoch * len(data_tr) + batch

                # print(self.epoch, self.batch, self.iter, data[0][0:5])

                self.set_mode(train=True)
                self.optimizer.zero_grad()
                data, targets = data.cuda(), targets.cuda()
                logits = self.model(data)
                B, T, C = logits.shape
                loss = self.criterion(logits.view(B*T, C), targets.view(B*T))
                self.train_loss.append(loss.item())

                loss.backward()
                self.optimizer.step()

                # stats
                do_subset = (self.iter + 1) % self.eval_subset_interval == 0
                do_full = (self.iter + 1) % self.eval_full_interval == 0
                evaluate = do_subset or do_full

                if evaluate:
                    should_save, val_time, test_time, computed_metrics = self.evaluate(all_data, do_full)
                    if should_save:
                        self.save('best')
                else:
                    should_save = False
                    val_time = 0
                    test_time = 0
                    computed_metrics = [(metric.abbrev, metric.default_str(), 0) for metric in self.custom_metrics]

                if (self.iter + 1) % (self.print_interval) == 0:
                    train_loss = np.mean(self.train_loss[-self.print_interval:])
                    val_loss = self.val_loss[-1]
                    test_loss = self.test_loss[-1]
                    self.print_line(evaluate, do_full, should_save, num_batches, train_loss, val_loss, test_loss, val_time, test_time, computed_metrics)

                if self.patience == 0:
                    self.logger.print('Stopping, out of patience')
                    return

    def evaluate(self, data, do_full):
        if do_full:
            data_val = data['val']
            data_test = data['test']
        else:
            data_val = subset(data['val'], subset_fraction=self.eval_subset_fraction)
            data_test = subset(data['test'], subset_fraction=self.eval_subset_fraction)

        t_val_start = time.time()
        val_loss = self.get_val_loss(data_val)
        val_time = time.time() - t_val_start

        t_test_start = time.time()
        test_loss = self.get_test_loss(data_test)
        test_time = time.time() - t_test_start

        self.eval_indices.append(self.iter)
        self.val_loss.append(val_loss)
        self.test_loss.append(test_loss)

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.reset_patience()
            should_save = True
        else:
            self.patience -= 1
            should_save = False

        computed_metrics = []
        for metric in self.custom_metrics:
            t_start = time.time()
            computed = metric.compute(self, data, {
                'train': self.train_loss,
                'val': self.val_loss,
                'test': self.test_loss,
            })
            computed_metrics.append((metric.abbrev, metric.format(computed), time.time() - t_start))

        return should_save, val_time, test_time, computed_metrics

    @torch.no_grad()
    def get_val_loss(self, data_val):
        self.set_mode(eval=True)
        val_loss = 0
        for data, targets in data_val:
            data, targets = data.cuda(), targets.cuda()
            logits = self.model(data)
            B, T, C = logits.shape
            loss = self.criterion(logits.view(B*T, C), targets.view(B*T))
            val_loss += loss.item() / len(data_val)
        return val_loss

    @torch.no_grad()
    def get_test_loss(self, data_test):
        self.set_mode(eval=True)
        test_loss = 0
        for data, targets in data_test:
            data, targets = data.cuda(), targets.cuda()
            logits = self.model(data)
            B, T, C = logits.shape
            loss = self.criterion(logits.view(B*T, C), targets.view(B*T))
            test_loss += loss.item() / len(data_test)
        return test_loss

    def format_time(self, t, nonempty):
        return

    def print_line(self, evaluated, full, saved, num_batches, train_loss, val_loss, test_loss, val_time, test_time, computed_metrics):
        full_str = 'F' if full else ' '
        save_str = 'S' if saved else ' '

        format_loss = lambda x: f'{x:6.4f}'
        format_time = lambda x: f'({x:4.1f}s)'
        omit_loss = ' ' * 6
        omit_time = ' ' * 7

        interval_time = time.time() - self.t_interval_end
        self.t_interval_end = time.time()
        total_time = self.t_interval_end - self.time_start
        metrics_time = sum([t for (_, _, t) in computed_metrics])
        train_time = interval_time - val_time - test_time - metrics_time

        train_loss_formatted = format_loss(train_loss)
        val_loss_formatted = format_loss(val_loss) if evaluated else omit_loss
        test_loss_formatted = format_loss(test_loss) if evaluated else omit_loss

        train_time_formatted = format_time(train_time)
        val_time_formatted = format_time(val_time) if evaluated else omit_time
        test_time_formatted = format_time(test_time) if evaluated else omit_time

        self.logger.print(f'[{full_str}]', end=' ')
        self.logger.print(f'I {self.iter+1:5d}', end=' ')
        self.logger.print(f'E {self.epoch+1:3d}/{self.num_epochs}', end=' ')
        self.logger.print(f'B {self.batch+1:4d}/{num_batches}', end=' ')
        self.logger.print(f'LR {self.lr:.4f}', end=' ')
        self.logger.print(f'TR {train_loss_formatted} {train_time_formatted}', end=' ')
        self.logger.print(f'[{save_str}]', end=' ')
        self.logger.print(f'VAL {val_loss_formatted} {val_time_formatted}', end=' ')
        self.logger.print(f'TEST {test_loss_formatted} {test_time_formatted}', end=' ')
        for (abbrev, computed, compute_time) in computed_metrics:
            compute_time_formatted = format_time(compute_time) if evaluated else omit_time
            self.logger.print(f'{abbrev} {computed} {compute_time_formatted}', end=' ')
        self.logger.print(f'TI {interval_time:5.1f}s, TT {total_time:6.1f}s')

    def visualize_loss(self):
        plt.plot(self.train_loss, color='lightgray', linewidth=0.3)
        train_loss_smoothed = np.convolve(self.train_loss, np.ones(self.eval_subset_interval)/self.eval_subset_interval, mode='same') # rolling avg, window equal to interval
        plt.plot(train_loss_smoothed)
        plt.plot(self.eval_indices, self.val_loss)
        plt.plot(self.eval_indices, self.test_loss)
        plt.axis([1, len(self.train_loss), 0, 8])
        plt.xlabel('Iter')
        plt.ylabel('Loss')
        plt.legend(['train', 'train_smoothed', 'val', 'test'])
        plt.show()
