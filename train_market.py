import logging
import os

import torch
from torch.optim import SGD, Adam, lr_scheduler

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip3 install tensorboardX")

from ignite.handlers import Timer
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

from attributer.opts import parse_opts
from attributer.model import generate_model
from attributer.dataset import get_data_market

from training.loss_utils import loss_market
from training.market_loss_metric import training_loss, testing_loss, cmc_metric
from training.market_utils import CosineMetric


def run(opt):
    if opt.log_file is not None:
        logging.basicConfig(filename=opt.log_file, level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    # logger.addHandler(logging.StreamHandler())
    logger = logger.info
    writer = SummaryWriter(log_dir=opt.log_dir)
    model_timer, data_timer = Timer(average=True), Timer(average=True)

    # Training variables
    logger('Loading models')
    model, parameters, mean, std = generate_model(opt)

    # Learning configurations
    if opt.optimizer == 'sgd':
        optimizer = SGD(parameters, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay,
                        nesterov=opt.nesterov)
    elif opt.optimizer == 'adam':
        optimizer = Adam(parameters, lr=opt.lr, betas=opt.betas)
    else:
        raise Exception("Not supported")
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=opt.lr_patience)

    # Loading checkpoint
    if opt.checkpoint:
        # load some param
        logger('loading checkpoint {}'.format(opt.checkpoint))
        checkpoint = torch.load(opt.checkpoint)

        # to use the loaded param
        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    logger('Loading dataset')  # =================================
    train_loader, val_loader, _ = get_data_market(opt, mean, std)

    device = 'cuda'
    trainer = create_supervised_trainer(model, optimizer,
                                        lambda pred, target: loss_market(pred, target, loss_fns=training_loss),
                                        device=device)
    evaluator = create_supervised_evaluator(model, metrics={
        'cosine_metric': CosineMetric(cmc_metric, testing_loss)}, device=device)

    # Training timer handlers
    model_timer.attach(trainer,
                       start=Events.EPOCH_STARTED,
                       resume=Events.ITERATION_STARTED,
                       pause=Events.ITERATION_COMPLETED,
                       step=Events.ITERATION_COMPLETED)
    data_timer.attach(trainer,

                      start=Events.EPOCH_STARTED,
                      resume=Events.ITERATION_COMPLETED,
                      pause=Events.ITERATION_STARTED,
                      step=Events.ITERATION_STARTED)

    # Training log/plot handlers
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % opt.log_interval == 0:
            logger("Epoch[{}] Iteration[{}/{}] Loss: {:.2f} Model Process: {:.3f}s/batch "
                   "Data Preparation: {:.3f}s/batch".format(engine.state.epoch, iter, len(train_loader),
                                                            engine.state.output, model_timer.value(),
                                                            data_timer.value()))
            writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    # Log/Plot Learning rate
    @trainer.on(Events.EPOCH_STARTED)
    def log_learning_rate(engine):
        lr = optimizer.param_groups[-1]['lr']
        logger('Epoch[{}] Starts with lr={}'.format(engine.state.epoch, lr))
        writer.add_scalar("learning_rate", lr, engine.state.epoch)

    # Checkpointing
    @trainer.on(Events.EPOCH_COMPLETED)
    def save_checkpoint(engine):
        if engine.state.epoch % opt.save_interval == 0:
            save_file_path = os.path.join(opt.result_path, 'save_{}.pth'.format(engine.state.epoch))
            states = {
                'epoch': engine.state.epoch,
                'arch': opt.model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)

    # val_evaluator event handlers
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics["cosine_metric"]
        # metric_values = [metrics[m] for m in val_metrics]
        logger("Validation Results - Epoch: {}".format(engine.state.epoch))
        for m, val in metrics.items():
            logger('{}: {:.4f}'.format(m, val))

        for m, val in metrics.items():
            if m in ['total_loss', 'cmc']:
                prefix = 'validation_summary/{}'
            else:
                prefix = 'validation/{}'
            writer.add_scalar(prefix.format(m), val, engine.state.epoch)

        # Update Learning Rate
        scheduler.step(metrics['cmc'])

    # kick everything off
    logger('Start training')
    trainer.run(train_loader, max_epochs=opt.n_epochs)

    writer.close()


if __name__ == "__main__":
    opt = parse_opts()

    run(opt)
