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
from attributer.attributes import WiderAttributes, get_attribute_names
from attributer.dataset import get_personattr_data, get_faceattr_data, get_tasks


from training.loss_utils import multitask_loss
from training.metric_utils import MultiAttributeMetric
from training.human_attr_loss_metric import get_losses_metrics, get_losses_metrics_fix
from training.metric_utils import print_summar_table


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def run(opt):
    if opt.log_file is not None:
        logging.basicConfig(filename=opt.log_file, level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    # logger.addHandler(logging.StreamHandler())
    logger = logger.info

    # Decide what tasks need to be performed given datasets
    tasks = get_tasks(opt)

    # Generate model based on tasks
    logger('Loading models')
    model, parameters, mean, std = generate_model(opt, tasks)
    # parameters[0]['lr'] = 0
    # parameters[1]['lr'] = opt.lr / 3

    logger('Loading dataset')
    if opt.option == 'person':
        attrs, train_loader, val_loader, _ = get_personattr_data(opt, tasks, mean, std)
    else:
        attrs, train_loader, val_loader, _ = get_faceattr_data(opt, tasks, mean, std)
    multi_dataset = len(opt.dataset.split(",")) > 1

    writer = create_summary_writer(model, train_loader, opt.log_dir)

    # Learning configurations
    if opt.optimizer == 'sgd':
        optimizer = SGD(parameters, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay,
                        nesterov=opt.nesterov)
    elif opt.optimizer == 'adam':
        optimizer = Adam(parameters, lr=opt.lr, betas=opt.betas)
    else:
        raise Exception("Not supported")
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=opt.lr_patience, factor=opt.factor)

    # Loading checkpoint
    if opt.checkpoint:
        logger('loading checkpoint {}'.format(opt.checkpoint))
        checkpoint = torch.load(opt.checkpoint)

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    device = 'cuda'
    attr_names = get_attribute_names(tasks, opt.specified_attrs)
    loss_fns, metrics = get_losses_metrics(tasks, opt.categorical_loss, opt.output_recognizable, opt.specified_attrs)
    trainer = create_supervised_trainer(model, optimizer,
                                        lambda pred, target: multitask_loss(pred, target, loss_fns=loss_fns),
                                        device=device)
    #attr_names = get_attribute_names(tasks, opt.specified_attrs)
    train_evaluator = create_supervised_evaluator(model, metrics={
        'multitask': MultiAttributeMetric(attr_names, metrics, tasks, opt.output_recognizable, opt.specified_attrs)}, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics={
        'multitask': MultiAttributeMetric(attr_names, metrics, tasks, opt.output_recognizable, opt.specified_attrs)}, device=device)

    # Training timer handlers
    model_timer, data_timer = Timer(average=True), Timer(average=True)
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
        data_list = [train_loader, val_loader]
        name_list = ['train', 'val']
        eval_list = [train_evaluator, val_evaluator]

        for data, name, evl in zip(data_list, name_list, eval_list):
            evl.run(data)
            metrics = evl.state.metrics["multitask"]
            logger(name + ": Validation Results - Epoch: {}".format(engine.state.epoch))

            for m, val in metrics['metrics'].items():
                #logger('{}: {:.4f}'.format(m, val))
                writer.add_scalar(name + '_metrics/{}'.format(m), val, engine.state.epoch)

            for m, val in metrics['summaries'].items():
                #logger('{}: {:.4f}'.format(m, val))
                writer.add_scalar(name + '_summary/{}'.format(m), val, engine.state.epoch)

            print_summar_table(opt.output_recognizable, logger, attr_names, metrics['logger'])

            # Update Learning Rate
            if name == 'val':
                scheduler.step(metrics['logger']['attr']['ap'][-1])

    # kick everything off
    logger('Start training')
    trainer.run(train_loader, max_epochs=opt.n_epochs)

    writer.close()


if __name__ == "__main__":
    opt = parse_opts()

    run(opt)
