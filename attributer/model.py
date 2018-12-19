from torch import nn
from attributer.all_in_one_person import AllInOnePerson
from attributer.all_in_one import AllInOne
from attributer.all_in_one_erised_person import AllInOneErisedPerson
from attributer.all_in_one_mot import AllInOneMOT
import pretrainedmodels


def conv_names():
    return pretrainedmodels.model_names


def generate_model(opt, tasks=None, inference=False): # opt
    assert opt.model in ['all_in_one', 'all_in_one_person', 'all_in_one_erised_person', 'all_in_one_mot']
    if opt.model == 'all_in_one':
        model = AllInOne(opt.conv, opt.pretrain)  # 'resnet18','False' , return a object of AllInOne
    elif opt.model == 'all_in_one_person':
        # TODO Just assume task == attribute right now. Will need to update
        assert tasks is not None
        model = AllInOnePerson(opt.conv, tasks, pretrained=opt.pretrain, img_size=opt.person_size,
                               attention=opt.attention, norm=opt.map_norm)
    elif opt.model == 'all_in_one_erised_person':
        model = AllInOneErisedPerson(opt.conv, tasks, pretrained=opt.pretrain, img_size=opt.person_size)
    elif opt.model == 'all_in_one_mot':
        model = AllInOneMOT(opt.conv, pretrained=opt.pretrain, img_size=(128, 64))
    else:
        raise Exception('Unsupported model {}'.format(opt.conv))
    mean, std = model.mean, model.std

    model = model.cuda()
    parameters = model.parameters()
    # parameters = model.get_parameter_groups()

    # Use multiple GPUs when training
    # if has more than one gpu
    if not inference:
        model = nn.DataParallel(model, device_ids=None)

    return model, parameters, mean, std