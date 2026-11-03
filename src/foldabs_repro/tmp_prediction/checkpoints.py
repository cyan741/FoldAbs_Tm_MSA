"""Persistent epoch archives, separate from rolling resume and best checkpoints."""
import math
import os
from pathlib import Path
import torch


def epoch_checkpoint_name(batch_size, learning_rate, seed, epoch):
    if batch_size < 1 or epoch < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Positive batch size, epoch and finite learning rate required')
    return f'bs{batch_size}_lr_{learning_rate:g}sd_{seed}_ep{epoch}.pt'


def nominal_batch_size(args, world_size):
    if getattr(args, 'loss_type', 'bce') == 'group_rank':
        return world_size * args.group_accumulation_steps * (1 + args.group_negatives)
    return world_size * args.gradient_accumulation


def archive_epoch(value, directory, *, batch_size, learning_rate, seed, epoch):
    """Publish without overwriting old epochs; repeated validation resumes are idempotent."""
    directory = Path(directory)
    destination = directory / epoch_checkpoint_name(batch_size, learning_rate, seed, epoch)
    payload = {**value, 'completed_epoch': epoch,
               'epoch_archive_batch_size': batch_size, 'epoch_archive_initial_lr': learning_rate,
               'epoch_archive_seed': seed}
    if destination.exists():
        previous = torch.load(destination, map_location='cpu', weights_only=False, mmap=True)
        for key in ('completed_epoch', 'step', 'epoch_archive_batch_size', 'epoch_archive_initial_lr', 'epoch_archive_seed'):
            if previous.get(key) != payload.get(key):
                raise FileExistsError(f'Conflicting epoch archive: {destination} ({key})')
        if previous['model'].keys() != payload['model'].keys() or any(
            not torch.equal(previous['model'][key], tensor.detach().cpu()) for key, tensor in payload['model'].items()
        ):
            raise FileExistsError(f'Conflicting epoch archive weights: {destination}')
        return destination
    temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.tmp')
    try:
        torch.save(payload, temporary)
        # Hard-link publication is atomic and refuses replacement, unlike rename.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
