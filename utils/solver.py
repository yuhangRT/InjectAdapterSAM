import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

from utils import logger
from utils.statics import AverageMeter, evaluator
from utils.sam_metrics import SAMMetrics, compute_cldice

__all__ = ['Trainer', 'Tester', 'SAMTrainer', 'SAMTester', 'SAMCriterion']


field = ('nmse', 'rho', 'epoch')
Result = namedtuple('Result', field, defaults=(None,) * len(field))


class Trainer:
    r""" The training pipeline for encoder-decoder architecture
    """

    def __init__(self, model, device, optimizer, criterion, scheduler, resume=None,
                 save_path='./checkpoints', print_freq=20, val_freq=10, test_freq=10):

        # Basic arguments
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

        # Verbose arguments
        self.resume_file = resume
        self.save_path = save_path
        self.print_freq = print_freq
        self.val_freq = val_freq
        self.test_freq = test_freq

        # Pipeline arguments
        self.cur_epoch = 1
        self.all_epoch = None
        self.train_loss = None
        self.val_loss = None
        self.test_loss = None
        self.best_rho = Result()
        self.best_nmse = Result()

        self.tester = Tester(model, device, criterion, print_freq)
        self.test_loader = None

    def loop(self, epochs, train_loader, val_loader, test_loader):
        r""" The main loop function which runs training and validation iteratively.

        Args:
            epochs (int): The total epoch for training
            train_loader (DataLoader): Data loader for training data.
            val_loader (DataLoader): Data loader for validation data.
            test_loader (DataLoader): Data loader for test data.
        """

        self.all_epoch = epochs
        self._resume()

        for ep in range(self.cur_epoch, epochs + 1):
            self.cur_epoch = ep

            # conduct training, validation and test
            self.train_loss = self.train(train_loader)
            if ep % self.val_freq == 0:
                self.val_loss = self.val(val_loader)

            if ep % self.test_freq == 0:
                self.test_loss, rho, nmse = self.test(test_loader)
            else:
                rho, nmse = None, None

            # conduct saving, visualization and log printing
            self._loop_postprocessing(rho, nmse)

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

        for batch_idx, (sparse_gt, ) in enumerate(data_loader):
            sparse_gt = sparse_gt.to(self.device)
            sparse_pred = self.model(sparse_gt)
            loss = self.criterion(sparse_pred, sparse_gt)

            # Scheduler update, backward pass and optimization
            if self.model.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            # Log and visdom update
            iter_loss.update(loss)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'Epoch: [{self.cur_epoch}/{self.all_epoch}]'
                            f'[{batch_idx + 1}/{len(data_loader)}] '
                            f'lr: {self.scheduler.get_lr()[0]:.2e} | '
                            f'MSE loss: {iter_loss.avg:.3e} | '
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

        if self.resume_file is None:
            return None
        assert os.path.isfile(self.resume_file)
        logger.info(f'=> loading checkpoint {self.resume_file}')
        checkpoint = torch.load(self.resume_file)
        self.cur_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_rho = checkpoint['best_rho']
        self.best_nmse = checkpoint['best_nmse']
        self.cur_epoch += 1  # start from the next epoch

        logger.info(f'=> successfully loaded checkpoint {self.resume_file} '
                    f'from epoch {checkpoint["epoch"]}.\n')

    def _loop_postprocessing(self, rho, nmse):
        r""" private function which makes loop() function neater.
        """

        # save state generate
        state = {
            'epoch': self.cur_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_rho': self.best_rho,
            'best_nmse': self.best_nmse
        }

        # save model with best rho and nmse
        if rho is not None:
            if self.best_rho.rho is None or self.best_rho.rho < rho:
                self.best_rho = Result(rho=rho, nmse=nmse, epoch=self.cur_epoch)
                state['best_rho'] = self.best_rho
                self._save(state, name=f"best_rho.pth")
            if self.best_nmse.nmse is None or self.best_nmse.nmse > nmse:
                self.best_nmse = Result(rho=rho, nmse=nmse, epoch=self.cur_epoch)
                state['best_nmse'] = self.best_nmse
                self._save(state, name=f"best_nmse.pth")

        self._save(state, name='last.pth')

        # print current best results
        if self.best_rho.rho is not None:
            print(f'\n=! Best rho: {self.best_rho.rho:.3e} ('
                  f'Corresponding nmse={self.best_rho.nmse:.3e}; '
                  f'epoch={self.best_rho.epoch})'
                  f'\n   Best NMSE: {self.best_nmse.nmse:.3e} ('
                  f'Corresponding rho={self.best_nmse.rho:.3e};  '
                  f'epoch={self.best_nmse.epoch})\n')


class Tester:
    r""" The testing interface for classification
    """

    def __init__(self, model, device, criterion, print_freq=20):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.print_freq = print_freq

    def __call__(self, test_data, verbose=True):
        r""" Runs the testing procedure.

        Args:
            test_data (DataLoader): Data loader for validation data.
        """

        self.model.eval()
        with torch.no_grad():
            loss, rho, nmse = self._iteration(test_data)
        if verbose:
            print(f'\n=> Test result: \nloss: {loss:.3e}'
                  f'    rho: {rho:.3e}    NMSE: {nmse:.3e}\n')
        return loss, rho, nmse

    def _iteration(self, data_loader):
        r""" protected function which test the model on given data loader for one epoch.
        """

        iter_rho = AverageMeter('Iter rho')
        iter_nmse = AverageMeter('Iter nmse')
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()

        for batch_idx, (sparse_gt, raw_gt) in enumerate(data_loader):
            sparse_gt = sparse_gt.to(self.device)
            sparse_pred = self.model(sparse_gt)
            loss = self.criterion(sparse_pred, sparse_gt)
            rho, nmse = evaluator(sparse_pred, sparse_gt, raw_gt)

            # Log and visdom update
            iter_loss.update(loss)
            iter_rho.update(rho)
            iter_nmse.update(nmse)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'[{batch_idx + 1}/{len(data_loader)}] '
                            f'loss: {iter_loss.avg:.3e} | rho: {iter_rho.avg:.3e} | '
                            f'NMSE: {iter_nmse.avg:.3e} | time: {iter_time.avg:.3f}')

        logger.info(f'=> Test rho:{iter_rho.avg:.3e}  NMSE: {iter_nmse.avg:.3e}\n')

        return iter_loss.avg, iter_rho.avg, iter_nmse.avg


# ============================= SAM-specific classes =============================


class SAMCriterion(nn.Module):
    """
    Combined loss for WireCR-SAM semantic segmentation.

    The loss operates on two foreground channels (wire and hole), with
    background handled implicitly during argmax decoding.
    """

    def __init__(
        self,
        num_classes=3,
        bce_weight=1.0,
        dice_weight=1.0,
        boundary_weight=0.0,
        cldice_weight=0.0,
        hole_class_weight=2.0,
    ):
        super(SAMCriterion, self).__init__()
        self.num_classes = num_classes
        self.foreground_classes = num_classes - 1
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.cldice_weight = cldice_weight
        class_weights = torch.ones(self.foreground_classes)
        if self.foreground_classes >= 2:
            class_weights[1] = hole_class_weight
        self.register_buffer("class_weights", class_weights)

    def _foreground_targets(self, targets):
        one_hot = F.one_hot(targets.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        return one_hot[:, 1:]

    def _dice_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        return 1 - dice.mean()

    def _boundary_map(self, tensor):
        dilated = F.max_pool2d(tensor, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-tensor, kernel_size=3, stride=1, padding=1)
        return (dilated - eroded).clamp(min=0.0, max=1.0)

    def forward(self, predictions, targets):
        class_logits = predictions["class_logits"]
        target_masks = targets["mask"].to(class_logits.device)
        foreground_targets = self._foreground_targets(target_masks)
        class_weights = self.class_weights.view(1, -1, 1, 1).to(class_logits.device)

        # Apply per-class positive weights manually to avoid BCE broadcasting
        # mismatches on dense NCHW segmentation logits.
        bce_map = F.binary_cross_entropy_with_logits(
            class_logits,
            foreground_targets,
            reduction="none",
        )
        positive_weight_map = 1.0 + (class_weights - 1.0) * foreground_targets
        bce_loss = (bce_map * positive_weight_map).mean()
        dice_loss = self._dice_loss(class_logits, foreground_targets)

        probs = torch.sigmoid(class_logits)
        boundary_loss = torch.tensor(0.0, device=class_logits.device)
        if self.boundary_weight > 0:
            pred_boundary = self._boundary_map(probs)
            target_boundary = self._boundary_map(foreground_targets)
            boundary_loss = F.l1_loss(pred_boundary, target_boundary)

        cldice_loss = torch.tensor(0.0, device=class_logits.device)
        if self.cldice_weight > 0 and self.foreground_classes > 0:
            wire_probs = probs[:, :1]
            wire_targets = foreground_targets[:, :1]
            cldice_loss = 1 - compute_cldice(wire_probs, wire_targets).mean()

        total_loss = (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
            + self.boundary_weight * boundary_loss
            + self.cldice_weight * cldice_loss
        )

        return {
            "loss": total_loss,
            "bce_loss": bce_loss.detach(),
            "dice_loss": dice_loss.detach(),
            "boundary_loss": boundary_loss.detach(),
            "cldice_loss": cldice_loss.detach(),
        }


class SAMTester:
    """Tester for WireCR-SAM semantic segmentation."""

    def __init__(self, model, device, criterion=None, print_freq=20):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.print_freq = print_freq
        self.metrics = SAMMetrics(
            num_classes=getattr(model, "num_classes", 3),
            class_names=getattr(model, "class_names", None),
        )

    def __call__(self, data_loader, verbose=True):
        """
        Run evaluation on the test set.

        Args:
            data_loader: DataLoader for test data
            verbose: Whether to print progress

        Returns:
            dict with evaluation results
        """
        self.model.eval()
        self.metrics.reset()

        iter_time = AverageMeter('Iter time')
        iter_loss = AverageMeter('Iter loss')
        time_tmp = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                images = batch["image"]
                gt_masks = batch["mask"].to(self.device)

                batched_input = [
                    {
                        "image": image,
                        "original_size": tuple(image.shape[-2:]),
                        "output_size": tuple(mask.shape[-2:]),
                    }
                    for image, mask in zip(images, gt_masks)
                ]

                outputs = self.model(batched_input, multimask_output=False)
                pred_masks = outputs["semantic_masks"]

                if self.criterion is not None:
                    loss_dict = self.criterion(outputs, {"mask": gt_masks})
                    iter_loss.update(loss_dict["loss"].item(), n=gt_masks.size(0))

                self.metrics.update(pred_masks, gt_masks)

                iter_time.update(time.time() - time_tmp)
                time_tmp = time.time()

                if verbose and (batch_idx + 1) % self.print_freq == 0:
                    current_metrics = self.metrics.compute()
                    logger.info(
                        f'[{batch_idx + 1}/{len(data_loader)}] '
                        f'IoU: {current_metrics["iou"]:.4f} | '
                        f'Dice: {current_metrics["dice"]:.4f} | '
                        f'BoundaryF1: {current_metrics["boundary_f1"]:.4f} | '
                        f'clDice: {current_metrics["cldice"]:.4f} | '
                        f'time: {iter_time.avg:.3f}'
                    )

        results = self.metrics.compute()
        results["loss"] = iter_loss.avg if iter_loss.count > 0 else 0.0

        if verbose:
            logger.info(f'\n=> SAM Test Results:')
            logger.info(f'  Loss: {results["loss"]:.4f}')
            logger.info(f'  IoU: {results["iou"]:.4f}')
            logger.info(f'  Dice: {results["dice"]:.4f}')
            logger.info(f'  Precision: {results["precision"]:.4f}')
            logger.info(f'  Recall: {results["recall"]:.4f}')
            logger.info(f'  F1: {results["f1"]:.4f}')
            logger.info(f'  BoundaryF1: {results["boundary_f1"]:.4f}')
            logger.info(f'  clDice: {results["cldice"]:.4f}')
            if "wire_iou" in results:
                logger.info(f'  Wire IoU: {results["wire_iou"]:.4f}')
            if "hole_iou" in results:
                logger.info(f'  Hole IoU: {results["hole_iou"]:.4f}')
                logger.info(f'  Hole Recall: {results["hole_recall"]:.4f}')
            if "foreground_iou" in results:
                logger.info(f'  Foreground IoU: {results["foreground_iou"]:.4f}')
            logger.info('')

        return results


class SAMTrainer:
    """Trainer for WireCR-SAM semantic segmentation."""

    def __init__(self, model, device, optimizer, criterion, scheduler,
                 save_path='./checkpoints', print_freq=20, val_freq=10, test_freq=10):

        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        # Verbose arguments
        self.save_path = save_path
        self.print_freq = print_freq
        self.val_freq = val_freq
        self.test_freq = test_freq

        # Pipeline state
        self.cur_epoch = 1
        self.all_epoch = None
        self.best_iou = 0.0

        # Initialize tester
        self.tester = SAMTester(model, device, criterion=criterion, print_freq=print_freq)

    def loop(self, epochs, train_loader, val_loader=None, test_loader=None):
        """
        Main training loop.

        Args:
            epochs: Total number of epochs to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            test_loader: Test data loader (optional)
        """
        self.all_epoch = epochs

        for ep in range(self.cur_epoch, epochs + 1):
            self.cur_epoch = ep

            # Training
            train_loss = self.train(train_loader)

            # Validation
            if val_loader is not None and ep % self.val_freq == 0:
                val_results = self.validate(val_loader)
            else:
                val_results = None

            # Testing
            if test_loader is not None and ep % self.test_freq == 0:
                test_results = self.tester(test_loader)
            else:
                test_results = None

            # Save checkpoint
            self._save_checkpoint(ep, train_loss, val_results, test_results)

            # Print progress
            self._print_progress(ep, train_loss, val_results, test_results)

    def train(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        iter_loss = AverageMeter('Loss')
        iter_time = AverageMeter('Time')
        time_tmp = time.time()

        for batch_idx, batch in enumerate(train_loader):
            gt_mask = batch["mask"].to(self.device)
            batched_input = [
                {
                    "image": image,
                    "original_size": tuple(image.shape[-2:]),
                    "output_size": tuple(mask.shape[-2:]),
                }
                for image, mask in zip(batch["image"], gt_mask)
            ]

            outputs = self.model(batched_input, multimask_output=False)
            loss_dict = self.criterion(outputs, {"mask": gt_mask})
            loss = loss_dict["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            iter_loss.update(loss.item(), n=gt_mask.size(0))
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(
                    f'Epoch: [{self.cur_epoch}/{self.all_epoch}] '
                    f'[{batch_idx + 1}/{len(train_loader)}] '
                    f'lr: {self.scheduler.get_lr()[0]:.2e} | '
                    f'loss: {iter_loss.avg:.3e} | '
                    f'bce: {loss_dict["bce_loss"].item():.3e} | '
                    f'dice: {loss_dict["dice_loss"].item():.3e} | '
                    f'time: {iter_time.avg:.3f}'
                )

        logger.info(f'=> Train Loss: {iter_loss.avg:.3e}\n')
        return iter_loss.avg

    def validate(self, val_loader):
        """Validate the model."""
        results = self.tester(val_loader, verbose=False)
        logger.info(
            f'=> Val IoU: {results["iou"]:.4f} | Dice: {results["dice"]:.4f} | '
            f'BoundaryF1: {results["boundary_f1"]:.4f} | clDice: {results["cldice"]:.4f}\n'
        )
        return results

    def _save_checkpoint(self, epoch, train_loss, val_results, test_results):
        """Save training checkpoint."""
        if self.save_path is None:
            return

        os.makedirs(self.save_path, exist_ok=True)

        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'best_iou': self.best_iou,
        }

        if val_results is not None:
            state['val_iou'] = val_results['iou']
            state['val_dice'] = val_results['dice']
            state['val_results'] = val_results

        if test_results is not None:
            state['test_iou'] = test_results['iou']
            state['test_dice'] = test_results['dice']
            state['test_results'] = test_results

            # Save best model
            if test_results['iou'] > self.best_iou:
                self.best_iou = test_results['iou']
                state['best_iou'] = self.best_iou
                torch.save(state, os.path.join(self.save_path, 'best_iou.pth'))
                logger.info(f'=> Saved best model with IoU: {self.best_iou:.4f}')

        state['best_iou'] = self.best_iou
        torch.save(state, os.path.join(self.save_path, 'last.pth'))

    def _print_progress(self, epoch, train_loss, val_results, test_results):
        """Print training progress."""
        if test_results is not None:
            logger.info(f'\n=! Best IoU: {self.best_iou:.4f} (current: {test_results["iou"]:.4f})\n')
