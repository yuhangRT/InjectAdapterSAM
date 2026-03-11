import time
import os
import torch
import torch.nn.functional as F
from collections import namedtuple

from utils import logger
from utils.statics import AverageMeter, evaluator
from utils.sam_metrics import SAMMetrics, compute_iou, compute_dice

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
    Combined loss for SAM+adapter training.

    Loss = mask_loss + iou_loss

    Args:
        mask_weight: Weight for mask loss
        iou_weight: Weight for IoU prediction loss
        dice_weight: Weight for Dice loss (optional)
    """

    def __init__(self, mask_weight=1.0, iou_weight=1.0, dice_weight=0.0):
        super(SAMCriterion, self).__init__()
        self.mask_weight = mask_weight
        self.iou_weight = iou_weight
        self.dice_weight = dice_weight

    def forward(self, predictions, targets):
        """
        Compute combined loss.

        Args:
            predictions: Dict with 'masks' and 'iou_scores'
            targets: Dict with 'masks' and 'iou_scores'

        Returns:
            Total loss
        """
        # Mask loss (binary cross-entropy with sigmoid)
        pred_masks = predictions['masks']  # (B, K, H, W)
        target_masks = targets['masks']  # (B, H, W)

        # Expand target to match number of predicted masks
        if pred_masks.dim() == 4:  # (B, K, H, W)
            target_masks = target_masks.unsqueeze(1).expand_as(pred_masks)

        mask_loss = F.binary_cross_entropy_with_logits(pred_masks, target_masks)

        # IoU prediction loss
        if 'iou_scores' in predictions and 'iou_scores' in targets:
            iou_loss = F.mse_loss(predictions['iou_scores'], targets['iou_scores'])
        else:
            iou_loss = torch.tensor(0.0, device=pred_masks.device)

        # Optional Dice loss
        dice_loss = torch.tensor(0.0, device=pred_masks.device)
        if self.dice_weight > 0:
            pred_sigmoid = torch.sigmoid(pred_masks)
            target = target_masks.float()
            intersection = (pred_sigmoid * target).sum(dim=(2, 3))
            union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
            dice = (2 * intersection) / (union + 1e-8)
            dice_loss = 1 - dice.mean()

        # Combine losses
        total_loss = (
            self.mask_weight * mask_loss +
            self.iou_weight * iou_loss +
            self.dice_weight * dice_loss
        )

        return total_loss


class SAMTester:
    """
    Tester for SAM with CRNet adapter.

    Evaluates segmentation quality using IoU, Dice, and other metrics.
    """

    def __init__(self, model, device, criterion=None, print_freq=20):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.print_freq = print_freq
        self.metrics = SAMMetrics()

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
        time_tmp = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # Move to device
                images = [item['image'].to(self.device) for item in batch]
                gt_masks = [item['ground_truth_mask'].to(self.device) for item in batch]
                original_sizes = [item['original_size'] for item in batch]
                prompts = [item['prompt'] for item in batch]

                # Prepare batched input for SAM
                batched_input = []
                for img, orig_size, prompt in zip(images, original_sizes, prompts):
                    input_dict = {
                        'image': img.cpu(),  # SAM handles preprocessing
                        'original_size': orig_size,
                    }
                    # Add prompts
                    if 'point_coords' in prompt:
                        input_dict['point_coords'] = prompt['point_coords']
                        input_dict['point_labels'] = prompt['point_labels']
                    if 'boxes' in prompt:
                        input_dict['boxes'] = prompt['boxes']

                    batched_input.append(input_dict)

                # Forward pass
                outputs = self.model(batched_input, multimask_output=True)

                # Process outputs and compute metrics
                for output, gt_mask in zip(outputs, gt_masks):
                    # Get best mask (highest IoU prediction)
                    pred_masks = output['masks']  # (K, H, W)
                    iou_preds = output['iou_predictions']  # (K,)

                    # Select best mask
                    best_idx = iou_preds.argmax()
                    best_mask = pred_masks[best_idx]

                    # Resize to ground truth size if needed
                    if best_mask.shape != gt_mask.shape:
                        best_mask = F.interpolate(
                            best_mask.unsqueeze(0).unsqueeze(0),
                            size=gt_mask.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()

                    # Convert to binary
                    best_mask_binary = (torch.sigmoid(best_mask) > 0.5).float()

                    # Update metrics
                    self.metrics.update(
                        best_mask_binary.unsqueeze(0),
                        gt_mask.unsqueeze(0)
                    )

                iter_time.update(time.time() - time_tmp)
                time_tmp = time.time()

                if verbose and (batch_idx + 1) % self.print_freq == 0:
                    current_metrics = self.metrics.compute()
                    logger.info(f'[{batch_idx + 1}/{len(data_loader)}] '
                               f'IoU: {current_metrics["iou"]:.4f} | '
                               f'Dice: {current_metrics["dice"]:.4f} | '
                               f'time: {iter_time.avg:.3f}')

        results = self.metrics.compute()

        if verbose:
            logger.info(f'\n=> SAM Test Results:')
            logger.info(f'  IoU: {results["iou"]:.4f}')
            logger.info(f'  Dice: {results["dice"]:.4f}')
            logger.info(f'  Precision: {results["precision"]:.4f}')
            logger.info(f'  Recall: {results["recall"]:.4f}')
            logger.info(f'  F1: {results["f1"]:.4f}\n')

        return results


class SAMTrainer:
    """
    Trainer for SAM with CRNet adapter.

    Handles training with multiple prompt strategies and tracks
    both segmentation metrics and loss.
    """

    def __init__(self, model, device, optimizer, criterion, scheduler,
                 num_prompts=1, prompt_strategy='random',
                 save_path='./checkpoints', print_freq=20, val_freq=10, test_freq=10):

        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        # Training configuration
        self.num_prompts = num_prompts
        self.prompt_strategy = prompt_strategy

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
        self.tester = SAMTester(model, device, print_freq=print_freq)

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
            # Process each sample (SAM processes one at a time)
            batch_loss = 0.0
            batch_count = 0

            for item in batch:
                # Move to device
                image = item['image'].to(self.device)
                gt_mask = item['ground_truth_mask'].to(self.device)
                original_size = item['original_size']
                prompt = item['prompt']

                # Prepare input
                batched_input = [{
                    'image': image.cpu(),
                    'original_size': original_size,
                }]

                # Add prompts
                if 'point_coords' in prompt:
                    batched_input[0]['point_coords'] = prompt['point_coords'].to(self.device)
                    batched_input[0]['point_labels'] = prompt['point_labels'].to(self.device)
                if 'boxes' in prompt:
                    batched_input[0]['boxes'] = prompt['boxes'].to(self.device)

                # Forward pass
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = self.model(batched_input, multimask_output=True)

                    # Prepare targets
                    pred_masks = outputs[0]['masks']  # (K, H, W)
                    pred_iou = outputs[0]['iou_predictions']  # (K,)

                    # Resize to match
                    if pred_masks.shape[-2:] != gt_mask.shape[-2:]:
                        pred_masks = F.interpolate(
                            pred_masks.unsqueeze(0),
                            size=gt_mask.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)

                    predictions = {
                        'masks': pred_masks,
                        'iou_scores': pred_iou,
                    }

                    # Compute IoU for target
                    with torch.no_grad():
                        best_iou = 0
                        for k in range(pred_masks.shape[0]):
                            pred_binary = (torch.sigmoid(pred_masks[k]) > 0.5).float()
                            iou = compute_iou(pred_binary.unsqueeze(0), gt_mask.unsqueeze(0))
                            best_iou = max(best_iou, iou.item())

                    targets = {
                        'masks': gt_mask,
                        'iou_scores': torch.tensor([best_iou], device=self.device),
                    }

                    # Compute loss
                    loss = self.criterion(predictions, targets)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                batch_loss += loss.item()
                batch_count += 1

            # Update metrics
            avg_loss = batch_loss / max(batch_count, 1)
            iter_loss.update(avg_loss)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'Epoch: [{self.cur_epoch}/{self.all_epoch}] '
                           f'[{batch_idx + 1}/{len(train_loader)}] '
                           f'lr: {self.scheduler.get_lr()[0]:.2e} | '
                           f'loss: {iter_loss.avg:.3e} | '
                           f'time: {iter_time.avg:.3f}')

        logger.info(f'=> Train Loss: {iter_loss.avg:.3e}\n')
        return iter_loss.avg

    def validate(self, val_loader):
        """Validate the model."""
        results = self.tester(val_loader, verbose=False)
        logger.info(f'=> Val IoU: {results["iou"]:.4f} | Dice: {results["dice"]:.4f}\n')
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

        if test_results is not None:
            state['test_iou'] = test_results['iou']
            state['test_dice'] = test_results['dice']

            # Save best model
            if test_results['iou'] > self.best_iou:
                self.best_iou = test_results['iou']
                torch.save(state, os.path.join(self.save_path, 'best_iou.pth'))
                logger.info(f'=> Saved best model with IoU: {self.best_iou:.4f}')

        torch.save(state, os.path.join(self.save_path, 'last.pth'))

    def _print_progress(self, epoch, train_loss, val_results, test_results):
        """Print training progress."""
        if test_results is not None:
            logger.info(f'\n=! Best IoU: {self.best_iou:.4f} (current: {test_results["iou"]:.4f})\n')
