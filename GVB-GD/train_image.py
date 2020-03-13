import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable
import random
import pdb
import math
from distutils.version import LooseVersion



def image_classification_test(loader, model, gvbg=False):
    start_test = True
    with torch.no_grad():
       iter_test = iter(loader["test"])
       for i in range(len(loader['test'])):
           data = iter_test.next()
           inputs = data[0]
           labels = data[1]
           inputs = inputs.cuda()
           labels = labels.cuda()
           _, outputs ,_  = model(inputs,gvbg=gvbg) 
           if start_test:
               all_output = outputs.float()
               all_label = labels.float()
               start_test = False
           else:
               all_output = torch.cat((all_output, outputs.float()), 0)
               all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy


def train(config):
    ## set pre-process
    prep_dict = {}
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_target(**config["prep"]['params'])
    prep_dict["target"] = prep.image_target(**config["prep"]['params'])
    prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
            shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
            shuffle=True, num_workers=4, drop_last=True)

    dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                            transform=prep_dict["test"])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                shuffle=False, num_workers=4)

    ## set base network
    class_num = config["network"]["params"]["class_num"]
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()

    ## add additional network for some methods
    ad_net = network.AdversarialNetwork( class_num, 1024)
    ad_net = ad_net.cuda()
 
    ## set optimizer
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    #multi gpu
    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i,k in enumerate(gpus)])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i,k in enumerate(gpus)])
    
    ## train   
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    for i in range(config["num_iterations"]):
        #test
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            temp_acc = image_classification_test(dset_loaders, base_network, gvbg=config["GVBG"])
            temp_model = nn.Sequential(base_network)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            print(log_str)
        #save model
        if i % config["snapshot_interval"] == 0:
            torch.save(base_network.state_dict(), osp.join(config["output_path"], \
                "iter_{:05d}_model.pth.tar".format(i)))

        ## train one iter
        base_network.train(True)
        ad_net.train(True)
        loss_params = config["loss"]                  
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        #dataloader
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
            
        #network
        inputs_source, labels_source = iter_source.next()
        inputs_target, _ = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        features_source, outputs_source, focal_source = base_network(inputs_source,gvbg=config["GVBG"])
        features_target, outputs_target, focal_target = base_network(inputs_target,gvbg=config["GVBG"])
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        focals = torch.cat((focal_source,focal_target),dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)

        #loss calculation
        transfer_loss, mean_entropy, gvbg, gvbd = loss.GVB([softmax_out,focals], ad_net, network.calc_coeff(i), GVBD=config['GVBD'])
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss + config["GVBG"] * gvbg + abs(config['GVBD']) * gvbd

        if i % config["print_num"] == 0:
            log_str = "iter: {:05d}, transferloss: {:.5f}, classifier_loss: {:.5f}, mean entropy:{:.5f}, gvbg:{:.5f}, gvbd:{:.5f}".format(i, transfer_loss, classifier_loss, mean_entropy, gvbg, gvbd)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            #print(log_str)

        total_loss.backward()
        optimizer.step()
    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc

if __name__ == "__main__":

    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), 'PyTorch>=1.0.0 is required'

    parser = argparse.ArgumentParser(description='Domain Adaptation')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', help="Options: ResNet50")
    parser.add_argument('--dset', type=str, default='office-home', help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='data/Art.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='data/Clipart.txt', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--print_num', type=int, default=100, help="interval of two print loss")
    parser.add_argument('--num_iterations', type=int, default=10002, help="interation num ")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--trade_off', type=float, default=1, help="parameter for transfer loss")
    parser.add_argument('--batch_size', type=int, default=36, help="batch size")
    parser.add_argument('--GVBG', type=float, default=0, help="lambda: parameter for GVBG (if lambda==0 then GVBG is not utilized)")
    parser.add_argument('--GVBD', type=float, default=0, help="mu: parameter for GVBD (if mu==0 then GVBD is not utilized)")
    parser.add_argument('--CDAN', type=bool, default=False, help="utilize CDAN or not)")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # train config
    config = {}
    config["GVBG"] = args.GVBG
    config["GVBD"] = args.GVBD 
    config["CDAN"] = args.CDAN
    config["gpu"] = args.gpu_id
    config["num_iterations"] = args.num_iterations 
    config["print_num"] = args.print_num
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = args.dset + "/" + args.output_dir

    if not osp.exists(config["output_path"]):
        os.system('mkdir -p '+config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])


    config["prep"] = {'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
    config["loss"] = {"trade_off":args.trade_off}
    if "ResNet" in args.net:
        config["network"] = {"name":network.ResNetFc, \
            "params":{"resnet_name":args.net, "use_bottleneck":False, "bottleneck_dim":256, "new_cls":True} }
    else:
        raise ValueError('Network cannot be recognized. Please define your own dataset here.')

    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }

    config["dataset"] = args.dset
    config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":args.batch_size}, \
                      "target":{"list_path":args.t_dset_path, "batch_size":args.batch_size}, \
                      "test":{"list_path":args.t_dset_path, "batch_size":args.batch_size}}

    if config["dataset"] == "office-home":
        seed = 2019
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 65
    elif config["dataset"] == "office":
        seed = 2019
        if   ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
             ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
             config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        elif ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
             ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
             ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
             ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
             config["optimizer"]["lr_param"]["lr"] = 0.0003 # optimal parameters
        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "visda":
        seed = 9297
        config["optimizer"]["lr_param"]["lr"] = 0.0003 # optimal parameters
        config["network"]["params"]["class_num"] = 12
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    print(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config["out_file"].write(str(config))
    config["out_file"].flush()
    train(config)
