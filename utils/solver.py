import time
import os
import torch
from collections import namedtuple
from scipy.io import savemat
from utils import logger
from utils.statics import AverageMeter, evaluator

__all__ = ['Trainer', 'Tester']


field = ('loss', 'epoch','mse','loss1')
Result = namedtuple('Result', field, defaults=(None,) * len(field))


class Trainer:
    r""" The training pipeline for encoder-decoder architecture
    """

    def __init__(self, model, device, optimizer, criterion,phase,resume=None,
                 save_path='./checkpoints', print_freq=20, val_freq=10, test_freq=10):

        # Basic arguments
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.phase = phase

        # Verbose arguments
        self.resume_file = resume
        self.save_path = save_path
        self.print_freq = print_freq
        self.test_freq = test_freq

        # Pipeline arguments
        self.cur_epoch = 1
        self.all_epoch = None
        self.train_loss = None
        self.val_loss = None
        self.test_loss = None
        self.best_loss = Result()
        self.best_mse = Result()

        self.tester = Tester(model, device, criterion,phase, print_freq)
        self.test_loader = None

    def loop(self, epochs, train_loader,test_loader):
        r""" The main loop function which runs training and validation iteratively.

        Args:
            epochs (int): The total epoch for training
            train_loader (DataLoader): Data loader for training data.
            val_loader (DataLoader): Data loader for validation data.
            test_loader (DataLoader): Data loader for test data.
        """

        self.all_epoch = epochs
        self._resume()
        if self.phase==1:
            for ep in range(self.cur_epoch, epochs + 1):
                self.cur_epoch = ep

                # conduct training, validation and test
                self.train_loss = self.train(train_loader)
                # if ep>2500 and ep<=2505:
                #     with torch.no_grad():
                #         for batch_idx, (h,) in enumerate(test_loader):
                #             if batch_idx==0:
                #                 sparse_pred = self.model(h)
                #                 ex_gt=torch.cat([ex_gt,sparse_pred],dim=0)
                # if ep == 2505:
                #     ex_gt=ex_gt.cpu().detach().numpy()
                #     savemat("X_AI.mat", {'X_AI':ex_gt})
                if ep % self.test_freq == 0:
                    self.test_loss = self.test(test_loader)
                    loss = self.test_loss
                else:
                    loss = None

                # conduct saving, visualization and log printing
                self._loop_postprocessing(loss)
        else:
            for ep in range(self.cur_epoch, epochs + 1):
                self.cur_epoch = ep

                # conduct training, validation and test
                self.train_loss = self.train(train_loader)
                # if ep>2500 and ep<=2505:
                #     with torch.no_grad():
                #         for batch_idx, (h,) in enumerate(test_loader):
                #             if batch_idx==0:
                #                 sparse_pred = self.model(h)
                #                 ex_gt=torch.cat([ex_gt,sparse_pred],dim=0)
                # if ep == 2505:
                #     ex_gt=ex_gt.cpu().detach().numpy()
                #     savemat("X_AI.mat", {'X_AI':ex_gt})
                if ep % self.test_freq == 0:
                    self.test_loss,loss1,loss2,loss3 = self.test(test_loader)
                    loss=self.test_loss
                else:
                    loss= None
                    loss1 = None
                    loss3 = None

                # conduct saving, visualization and log printing
                self._loop_postprocessing1(loss,loss1,loss3)

    def train(self, train_loader):
        r""" train the model on the given data loader for one epoch.

        Args:
            train_loader (DataLoader): the training data loader
        """

        self.model.train()
        with torch.enable_grad():
            return self._iteration(train_loader)

    def val(self, val_loader):
        r""" exam the model with validation set.

        Args:
            val_loader: (DataLoader): the validation data loader
        """

        self.model.eval()
        with torch.no_grad():
            return self._iteration(val_loader)

    def test(self, test_loader):
        r""" Truly test the model on the test dataset for one epoch.

        Args:
            test_loader (DataLoader): the test data loader
        """

        self.model.eval()
        with torch.no_grad():
            return self.tester(test_loader, verbose=False)

    def _iteration(self, data_loader):
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()
        if self.phase==1:
            for batch_idx, (h, ) in enumerate(data_loader):
                h = h.to(self.device)
                x_pred = self.model(h)
                loss = self.criterion(h,x_pred)
                
                # Scheduler update, backward pass and optimization
                if self.model.training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Log and visdom update
                iter_loss.update(loss)
                iter_time.update(time.time() - time_tmp)
                time_tmp = time.time()

                # plot progress
                if (batch_idx + 1) % self.print_freq == 0:
                    logger.info(f'Epoch: [{self.cur_epoch}/{self.all_epoch}]'
                                f'[{batch_idx + 1}/{len(data_loader)}] '
                                f'Balance loss: {iter_loss.avg:.3e} | '
                                f'time: {iter_time.avg:.3f}')

            mode = 'Train' if self.model.training else 'Val'
            logger.info(f'=> {mode}  Loss: {iter_loss.avg:.3e}\n')
        else:
            for batch_idx, (h1,h2, ) in enumerate(data_loader):
                h1 = h1.to(self.device)
                h2 = h2.to(self.device)
                f1,f2 = self.model(h1,h2)
                loss,loss1,loss2,loss3 = self.criterion(h1,h2,f1,f2)
                
                # Scheduler update, backward pass and optimization
                if self.model.training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Log and visdom update
                iter_loss.update(loss)
                iter_time.update(time.time() - time_tmp)
                time_tmp = time.time()

                # plot progress
                if (batch_idx + 1) % self.print_freq == 0:
                    logger.info(f'Epoch: [{self.cur_epoch}/{self.all_epoch}]'
                                f'[{batch_idx + 1}/{len(data_loader)}] '
                                f'Balance loss: {iter_loss.avg:.3e} | '
                                f'time: {iter_time.avg:.3f}')

            mode = 'Train' if self.model.training else 'Val'
            logger.info(f'=> {mode}  Loss: {iter_loss.avg:.3e}\n')
        return iter_loss.avg

    def _save(self, state, name):
        if self.save_path is None:
            logger.warning('No path to save checkpoints.')
            return

        os.makedirs(self.save_path, exist_ok=True)
        torch.save(state, os.path.join(self.save_path, name))

    def _resume(self):
        r""" protected function which resume from checkpoint at the beginning of training.
        """
        if self.phase==1:
            if self.resume_file is None:
                return None
            assert os.path.isfile(self.resume_file)
            logger.info(f'=> loading checkpoint {self.resume_file}')
            checkpoint = torch.load(self.resume_file)
            self.cur_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            # self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.best_loss = checkpoint['best_loss_p1']
            self.cur_epoch += 1  # start from the next epoch

            logger.info(f'=> successfully loaded checkpoint {self.resume_file} '
                        f'from epoch {checkpoint["epoch"]}.\n')
        else:
            if self.resume_file is None:
                return None
            assert os.path.isfile(self.resume_file)
            logger.info(f'=> loading checkpoint {self.resume_file}')
            checkpoint = torch.load(self.resume_file)
            self.cur_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            # self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.best_loss = checkpoint['best_loss']
            self.best_mse = checkpoint['best_mse']
            self.cur_epoch += 1  # start from the next epoch

            logger.info(f'=> successfully loaded checkpoint {self.resume_file} '
                        f'from epoch {checkpoint["epoch"]}.\n')

    def _loop_postprocessing(self, loss):
        r""" private function which makes loop() function neater.
        """

        # save state generate
        state = {
            'epoch': self.cur_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_loss': self.best_loss
        }

        # save model with best loss
        if loss is not None:
            if self.best_loss.loss is None or self.best_loss.loss > loss:
                self.best_loss = Result(loss=loss, epoch=self.cur_epoch)
                state['best_loss'] = self.best_loss
                self._save(state, name=f"best_loss_p1.pth")

        self._save(state, name='last.pth')

        # print current best results
        if self.best_loss.loss is not None:
            print(f'\n=! Best loss: {self.best_loss.loss:.3e} '
                  f'epoch={self.best_loss.epoch})')
    def _loop_postprocessing1(self, loss,loss1,loss3):
        r""" private function which makes loop() function neater.
        """

        # save state generate
        state = {
            'epoch': self.cur_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'best_mse':self.best_mse
        }

        # save model with best loss
        if loss is not None:
            if self.best_loss.loss is None or self.best_loss.loss > loss:
                self.best_loss = Result(loss=loss,mse=loss3, loss1=loss1,epoch=self.cur_epoch)
                state['best_loss'] = self.best_loss
                self._save(state, name=f"best_loss_p2.pth")
        if loss3 is not None:
            if self.best_mse.mse is None or self.best_mse.mse > loss3:
                self.best_mse = Result(mse=loss3, loss=loss, loss1=loss1,epoch=self.cur_epoch)
                state['best_mse'] = self.best_mse
                self._save(state, name=f"best_mse.pth")

        self._save(state, name='last.pth')

        # print current best results
        if self.best_loss.loss is not None:
            print(f'\n=! Best loss: {self.best_loss.loss:.3e} '
                  f'epoch={self.best_loss.epoch})'
                  f'corresponding mse={self.best_loss.mse})'
                  f'corresponding loss1={self.best_loss.loss1})')        
        if self.best_mse.mse is not None:
            print(f'\n=! Best mse: {self.best_mse.mse:.3e} '
                  f'epoch={self.best_mse.epoch})'
                  f'corresponding loss={self.best_mse.loss})'
                  f'corresponding loss1={self.best_mse.loss1})')


class Tester:
    r""" The testing interface for classification
    """

    def __init__(self, model, device, criterion, phase,print_freq=10):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.phase=phase
        self.print_freq = print_freq

    def __call__(self, test_data, verbose=True):
        r""" Runs the testing procedure.

        Args:
            test_data (DataLoader): Data loader for validation data.
        """

        self.model.eval()
        with torch.no_grad():
            loss = self._iteration(test_data)
        if verbose:
            print(f'\n=> Test result: \nloss: {loss:.3e}')
        return loss
    
    def _iteration(self, data_loader):
        r""" protected function which test the model on given data loader for one epoch.
        """
        if self.phase==1:
            iter_loss = AverageMeter('Iter loss')
            iter_time = AverageMeter('Iter time')
            time_tmp = time.time()

            for batch_idx, (h,) in enumerate(data_loader):
                h = h.to(self.device)
                x_pred = self.model(h,)
                loss = self.criterion(h,x_pred)
                # Log and visdom update
                iter_loss.update(loss)
                iter_time.update(time.time() - time_tmp)
                time_tmp = time.time()

                # plot progress
                if (batch_idx + 1) % self.print_freq == 0:
                    logger.info(f'[{batch_idx + 1}/{len(data_loader)}] '
                                f'loss: {iter_loss.avg:.3e} | time: {iter_time.avg:.3f}')

            logger.info(f'=> Test loss:{iter_loss.avg:.3e} \n')
            return iter_loss.avg
        else:
            iter_loss = AverageMeter('Iter loss')
            iter_loss1 = AverageMeter('Iter loss1')
            iter_loss2 = AverageMeter('Iter loss2')
            iter_loss3 = AverageMeter('Iter loss3')
            iter_time = AverageMeter('Iter time')
            time_tmp = time.time()

            for batch_idx, (h1,h2,) in enumerate(data_loader):
                h1 = h1.to(self.device)
                h2 = h2.to(self.device)
                f1,f2 = self.model(h1,h2)
                loss,loss1,loss2,loss3 = self.criterion(h1,h2,f1,f2)
                # Log and visdom update
                iter_loss.update(loss)
                iter_loss1.update(loss1)
                iter_loss2.update(loss2)
                iter_loss3.update(loss3)
                iter_time.update(time.time() - time_tmp)
                time_tmp = time.time()

                # plot progress
                if (batch_idx + 1) % self.print_freq == 0:
                    logger.info(f'[{batch_idx + 1}/{len(data_loader)}] '
                                f'loss: {iter_loss.avg:.3e} | time: {iter_time.avg:.3f}'
                                f'loss1: {iter_loss1.avg:.3e} | time: {iter_time.avg:.3f}'
                                f'loss2: {iter_loss2.avg:.3e} | time: {iter_time.avg:.3f}'
                                f'loss3: {iter_loss3.avg:.3e} | time: {iter_time.avg:.3f}')

            logger.info(f'=> Test loss:{iter_loss.avg:.3e} loss1: {iter_loss1.avg:.3e} loss2: {iter_loss2.avg:.3e} loss3: {iter_loss3.avg:.3e}\n')

            return iter_loss.avg,iter_loss1.avg,iter_loss2.avg,iter_loss3.avg