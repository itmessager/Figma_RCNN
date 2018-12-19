from ignite.metrics import CategoricalAccuracy, Loss
from attributer.attributes import WiderAttributes as WA
from attributer.attributes import ErisedAttributes as EA
from training.metric_utils import AveragePrecision
from training.loss_utils import get_categorial_loss, reverse_ohem_loss
from attributer.attributes import AttributeType


def get_losses_metrics(attrs, categorical_loss='cross_entropy', output_recognizable=False, specified_attrs=[]):
    loss_fn = get_categorial_loss(categorical_loss)

    losses = []
    metrics = []
    if not specified_attrs:
        specified_attrs = [attr.name for attr in attrs]
    for attr in attrs:
        # For attribute classification
        losses.append(loss_fn)

        if attr.name in specified_attrs:
            if attr.data_type == AttributeType.BINARY:
                metrics.append([AveragePrecision(), CategoricalAccuracy(), Loss(loss_fn)])
            elif attr.data_type == AttributeType.MULTICLASS:
                metrics.append([CategoricalAccuracy(), Loss(loss_fn)])
            elif attr.data_type == AttributeType.NUMERICAL:
                # not support now
                pass

            # For recognizability classification
            if output_recognizable:
                # TODO Reverse AP only works well for WiderAttribute
                metrics.append([AveragePrecision(reverse=True), CategoricalAccuracy(), Loss(reverse_ohem_loss)])

        if output_recognizable:
            losses.append(reverse_ohem_loss)  # Always use reverse OHEM loss for recognizability, at least for now

    return losses, metrics

#
# def get_losses_metrics(categorical_loss='cross_entropy', output_recognizable=False):
#     loss_fn = get_categorial_loss(categorical_loss)
#
#     losses = []
#     metrics = []
#     for _ in WA:
#         # For attribute classification
#         losses.append(loss_fn)
#         #metrics.append([AveragePrecision(), CategoricalAccuracy(), Loss(loss_fn)])
#         metrics.append([AveragePrecision(), CategoricalAccuracy(), Loss(loss_fn)])
#
#         # For recognizability classification
#         if output_recognizable:
#             losses.append(reverse_ohem_loss)  # Always use reverse OHEM loss for recognizability, at least for now
#             # TODO Reverse AP only works well for WiderAttribute
#             metrics.append([AveragePrecision(reverse=True), CategoricalAccuracy(), Loss(reverse_ohem_loss)])
#
#     return losses, metrics


def get_losses_metrics_fix(categorical_loss='cross_entropy', output_recognizable=False):
    loss_fn = get_categorial_loss(categorical_loss)

    losses = []
    metrics = []
    for attr in EA:
        losses.append(loss_fn)
        if str(attr) in ["glasses", "pregnant", "tottoo", "carry", "sex"]:
            metrics.append([AveragePrecision(reverse=True), CategoricalAccuracy(), Loss(loss_fn)])
        else:
            metrics.append([CategoricalAccuracy(), Loss(loss_fn)])

        if output_recognizable:
            losses.append(loss_fn)
            metrics.append([AveragePrecision(reverse=True), CategoricalAccuracy(), Loss(loss_fn)])

    return losses, metrics

