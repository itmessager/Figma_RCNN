import torch
import torch.nn.functional as F
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric, CategoricalAccuracy, Loss
from training.loss_utils import select_samples_by_mask
from abc import abstractmethod
from sklearn.metrics import average_precision_score
import collections
import math



# TODO Remove this function as ignite provides out-of-box mAP calculation
class EpochMetric(Metric):
    _predictions, _targets = None, None

    def reset(self):
        self._predictions = torch.tensor([], dtype=torch.float32)
        self._targets = torch.tensor([], dtype=torch.long)

    def update(self, output):
        y_pred, y = output

        assert 1 <= y_pred.ndimension() <= 2, "Predictions should be of shape (batch_size, n_classes)"
        assert 1 <= y.ndimension() <= 2, "Targets should be of shape (batch_size, n_classes)"

        if y.ndimension() == 2:
            assert torch.equal(y ** 2, y), 'Targets should be binary (0 or 1)'

        if y_pred.ndimension() == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(dim=-1)

        if y.ndimension() == 2 and y.shape[1] == 1:
            y = y.squeeze(dim=-1)

        y_pred = y_pred.type_as(self._predictions)
        y = y.type_as(self._targets)

        self._predictions = torch.cat([self._predictions, y_pred], dim=0)
        self._targets = torch.cat([self._targets, y], dim=0)

    @abstractmethod
    def compute(self):
        pass


class AveragePrecision(EpochMetric):
    def __init__(self, reverse=False):
        super().__init__()
        self.reverse = reverse

    def compute(self):
        y_true = self._targets.numpy()
        y_pred = F.softmax(self._predictions, 1).numpy()
        if not self.reverse:
            return average_precision_score(y_true, y_pred[:, 1])
        else:
            # Treat negative example as positive and vice-verse, to calculate AP
            return average_precision_score(1 - y_true, y_pred[:, 0])


def print_metric_name(metric):
    if isinstance(metric, AveragePrecision):
        return 'ap'
    elif isinstance(metric, CategoricalAccuracy):
        return 'accuracy'
    elif isinstance(metric, Loss):
        return 'loss'
    else:
        return metric.__class__.__name__.lower()


def print_summar_table(output_recognizable, logger, attr_display_names, logger_info):
    attr_title = ['{:^12}'.format('attr')]
    recognizable_title = ['{:^12}'.format('recognizable')]
    for names in attr_display_names:
        if names.endswith('recognizable'):
            recognizable_title.append('{:^10}'.format(names.split('/')[0]))
        else:
            attr_title.append('{:^10}'.format(names))
    attr_title.append('{:^10}'.format('summary'))
    recognizable_title.append('{:^10}'.format('summary'))

    # attr
    logger(attr_title)
    name_list = ['ap', 'accuracy', 'loss']
    print_info = {}
    for name in name_list:
        print_info[name] = ['{:^12}'.format(name)]
    for key, item in logger_info['attr'].items():
        for val in item:
            print_info[key].append('{:^10}'.format(val))
    for name in name_list:
        logger(print_info[name])

    # detectability
    if output_recognizable:
        for name in name_list:
            print_info[name] = ['{:^12}'.format(name)]
        logger('\n')
        logger(recognizable_title)
        for key, item in logger_info['detect'].items():
            for val in item:
                print_info[key].append('{:^10}'.format(val))
        for name in name_list:
            logger(print_info[name])


class TableForPrint(object):
    def __init__(self, output_recognizable):
        self.logger_print, self.metrics, self.summary = self.create()
        self.output_recognizable = output_recognizable

    # for every new attr, add 0 for placeholder
    def reset(self, name):
        if not name.endswith('/recognizable'):
            for _, attr_detect in self.logger_print.items():
                for _, ap_acc_loss in attr_detect.items():
                    ap_acc_loss.append('None')

    def update(self, attr_detect, ap_acc_loss, val):
        if attr_detect.endswith('/recognizable'):
            self.logger_print['detect'][ap_acc_loss][-1] = float('{:^10.4f}'.format(val))
        else:
            self.logger_print['attr'][ap_acc_loss][-1] = float('{:^10.4f}'.format(val))
        metrics_name = attr_detect + '/' + ap_acc_loss
        self.metrics[metrics_name] = val

    def summarize(self):
        if not self.output_recognizable:
            name_list = ['attr']
        else:
            name_list = ['attr', 'detect']
        fil = {}
        for name in name_list:
            for key, item in self.logger_print[name].items():
                fil[key] = list(filter(lambda x: x is not 'None' and not math.isnan(x), item))
            ap = float('{:^10.4f}'.format(sum(fil['ap']) / len(fil['ap'])))
            accuracy = float('{:^10.4f}'.format(sum(fil['accuracy']) / len(fil['accuracy'])))
            loss = float('{:^10.4f}'.format(sum(fil['loss'])))
            for evl, evl_name in zip(['ap', 'accuracy', 'loss'], ['mAP', 'mAccuracy', 'total_loss']):
                self.logger_print[name][evl].append(locals()[evl])
                summary_name = name + '/' + evl_name
                #if name == 'detect':
                self.summary[summary_name] = locals()[evl]
                if summary_name == 'detect/total_loss':
                    self.summary['total_loss'] = self.summary['attr/total_loss'] + self.summary['detect/total_loss']

    def create(self):
        logger_for_print = collections.OrderedDict()
        name_list = ['ap', 'accuracy', 'loss']
        for name in name_list:
            logger_for_print[name] = []

        logger_for_print_detect = collections.OrderedDict()
        for name in name_list:
            logger_for_print_detect[name] = []

        logger_print = collections.OrderedDict()
        logger_print['attr'] = logger_for_print
        logger_print['detect'] = logger_for_print_detect

        metrics = {}
        summary = {}
        return logger_print, metrics, summary


class MultiAttributeMetric(Metric):
    def __init__(self, names, metrics_per_attr, attrs, output_recognizable, specified_attrs=[]):
        self.names = names
        self.output_recognizable = output_recognizable
        self.metrics_per_attr = [ma if isinstance(ma, list) else [ma] for ma in metrics_per_attr]
        self.attr_numbers = self.get_attr_numbers(attrs, output_recognizable, specified_attrs)

        super().__init__()

    def reset(self):
        for ma in self.metrics_per_attr:
            for m in ma:
                m.reset()

    def update(self, output):
        preds, (target, mask) = output

        n_samples = target[0].size()[0]
        n_tasks = len(target)
        index = torch.arange(n_samples, dtype=torch.long, device='cuda')

        for i, j in zip(self.attr_numbers, range(len(self.attr_numbers))):
                if mask[i].any():
                    pred, gt = select_samples_by_mask(preds[i], target[i], mask[i], index)
                    for m in self.metrics_per_attr[j]:
                        m.update((pred, gt))

    def compute(self):
        metrics = {}
        summaries = {}

        # logger_print = create_orderdict_for_print()
        table = TableForPrint(self.output_recognizable)

        # for each Attribute:
        for name, ma in zip(self.names, self.metrics_per_attr):
            table.reset(name)
            # Set metric display name
            for m in ma:
                m_name = print_metric_name(m)
                # metric_name = name + "/" + print_metric_name(m)
                try:
                    # metrics[metric_name] = m.compute()
                    table.update(name, m_name, m.compute())
                except NotComputableError:
                    continue  # There hasn't been any sample that contains this attribute yet

        table.summarize()
        # Compute total_loss if necessary
        # # TODO What if weights on losses need to be supported
        # losses = [value for key, value in metrics.items() if key.endswith('loss')]
        # if len(losses) > 0:
        #     summaries['total_loss'] = sum(losses)
        #
        # def get_statistics_summary(stat_name, metric_suffix):
        #     # Calculate sub-task (such as recognizability) statistics separately.
        #     stats = {}
        #     for key, value in metrics.items():
        #         if key.endswith(metric_suffix):
        #             # So value of 'jeans/ap' will be added to stats['']
        #             # while 'jeans/recognizability/ap' will be added to stats['recognizability/']
        #             prefix = key[key.find('/') + 1:key.rfind('/') + 1]
        #             if prefix not in stats:
        #                 stats[prefix] = [value]
        #             else:
        #                 stats[prefix].append(value)
        #     if len(stats) > 0:
        #         for prefix, values in stats.items():
        #             summaries[prefix + stat_name] = sum(values) / len(values)
        #
        # # Compute mAP if AP is used. Calculate sub-task (such as recognizability) statistics separately.
        # get_statistics_summary("mAP", '/ap')
        #
        # # Compute average accuracy if accuracy is used
        # get_statistics_summary("mAccuracy", 'accuracy')
        # aps = {}
        # for key, value in metrics.items():
        #     if key.endswith('/ap'):
        #         prefix = key[key.find('/')+1:key.rfind('/')+1]
        #         if prefix not in aps:
        #             aps[prefix] = [value]
        #         else:
        #             aps[prefix].append(value)
        # if len(aps) > 0:
        #     for prefix, values in aps.items():
        #         summaries[prefix + "mAP"] = sum(values) / len(values)

        # Compute average accuracy if accuracy is used
        # accuracies = [value for key, value in metrics.items() if key.endswith('accuracy')]
        # if len(accuracies) > 0:
        #     summaries["mAccuracy"] = sum(accuracies) / len(accuracies)
        return {'metrics': table.metrics, 'summaries': table.summary, 'logger': table.logger_print}

    def get_attr_numbers(self, attrs, output_recognizable, specified_attrs):
        attr_numbers = []
        if not specified_attrs:
            specified_attrs = [attr.name for attr in attrs]
        for i in range(len(attrs)):
            if attrs[i].name in specified_attrs:
                if output_recognizable:
                    attr_numbers.append(2*i)
                    attr_numbers.append(2*i + 1)
                else:
                    attr_numbers.append(i)
        return attr_numbers


# Utility Metric to return a scaled output of the actual metric
class ScaledError(Metric):
    def __init__(self, metric, scale=1.0):
        self.metric = metric
        self.scale = scale

        super().__init__()

    def reset(self):
        self.metric.reset()

    def update(self, output):
        self.metric.update(output)

    def compute(self):
        return self.scale * self.metric.compute()
