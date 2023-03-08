import torch
from torch.autograd import Variable
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score

import importlib
import loss_fn
import models

importlib.reload(models)
importlib.reload(loss_fn)
import copy

ce_loss = nn.CrossEntropyLoss()
softmax = torch.nn.Softmax(dim=1)
f_sm = nn.Softmax(dim=1)

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams["figure.dpi"] = 100
matplotlib.rcParams["figure.dpi"] = 300
plt.style.use('bmh')
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
legend_properties = {'weight': 'bold', 'size': 10}


def cal_acc(pred, label):
    '''

    :param pred: prediction from the model, list of tensor
    :param label: true label, list of tensor
    :return: accuracy
    '''
    pred_t = torch.concat(pred)
    prediction = torch.argmax(pred_t, dim=-1).unsqueeze(-1)
    label_t = torch.concat(label)
    acc = (prediction == label_t).sum() / len(pred_t)
    return acc


def cal_pos_acc(pred, label, pos_ind):
    '''

    :param pred: prediction from the model, list of tensor
    :param label: true label, list of tensor
    :param pos_ind: positive instance index
    :return:  accuracy of positive instances
    '''
    pred_t = torch.concat(pred)
    prediction = torch.argmax(pred_t, dim=-1).unsqueeze(-1)
    label_t = torch.concat(label)
    # positive index
    ind = [i for i in range(len(pred_t)) if label_t[i] == pos_ind]
    acc = (prediction[ind] == label_t[ind]).sum() / len(ind)
    return acc


def train_model(args, c_fold, model, model_opt, train_dataloader, dev_dataloader, ce_loss, ce_val_loss):
    # train the model and return the best model
    # best model is based on auroc
    patience = 0
    for j in range(args.epochs):
        model.train()
        sofa_list = []
        sofap_list = []
        loss_t = []
        loss_to = []

        for vitals, target, key_mask in train_dataloader:
            # print(label.shape)
            if args.warmup == True:
                model_opt.optimizer.zero_grad()
            else:
                model_opt.zero_grad()
            # ti_data = Variable(ti.float().to(device))
            td_data = vitals.to(device)  # (6, 182, 24)
            sofa = target.to(device)  # (6, )
            key_mask = key_mask.to(device)
            # tgt_mask = model.get_tgt_mask(td_data.shape[-1]).to(device)
            # (16, 80, 2)
            if args.model_name == 'TCN':

                sofa_p = model(td_data)
            elif args.model_name == 'RNN':
                # x_lengths have to be a 1d tensor
                td_transpose = td_data.transpose(1, 2)
                x_lengths = torch.LongTensor([len(key_mask[i][key_mask[i] == 0]) for i in range(key_mask.shape[0])])
                sofa_p = model(td_transpose, x_lengths)
            else:
                tgt_mask = model.get_tgt_mask(vitals.to(device).shape[-1]).to(device)
                sofa_p = model(vitals.to(device), tgt_mask, key_mask.to(device))
                # if seq_handel = 'scale', sofa_p
            # if seq_handel = 'scale':
            # else:
            if args.loss_rule == 'mean':
                pred = torch.stack([sofa_p[i][key_mask[i] == 0].mean(dim=-2) for i in range(len(sofa_p))])
            elif args.loss_rule == 'last':
                pred = torch.stack([sofa_p[i][key_mask[i] == 0][-1, :] for i in range(len(sofa_p))])
            loss = ce_loss(pred, sofa.squeeze(-1))
            loss.backward()
            model_opt.step()

            sofa_list.append(sofa)
            sofap_list.append(pred)
            loss_t.append(loss)

        train_acc = cal_acc(sofap_list, sofa_list)
        print('Train acc is %.2f%%' % (train_acc * 100))

        loss_avg = np.mean(torch.stack(loss_t, dim=0).cpu().detach().numpy())

        model.eval()
        y_list = []
        y_pred_list = []
        ti_list = []
        td_list = []
        loss_val = []
        with torch.no_grad():  # validation does not require gradient

            for vitals, target, key_mask in dev_dataloader:
                # ti_test = Variable(torch.FloatTensor(ti)).to(device)
                td_test = Variable(vitals.float().to(device))
                sofa_t = Variable(target.long().to(device))
                key_mask = key_mask.to(device)

                if args.model_name == 'TCN':

                    sofap_t = model(td_test)
                elif args.model_name == 'RNN':
                    # x_lengths have to be a 1d tensor
                    td_transpose = td_test.transpose(1, 2)
                    x_lengths = torch.LongTensor([len(key_mask[i][key_mask[i] == 0]) for i in range(key_mask.shape[0])])
                    sofap_t = model(td_transpose, x_lengths)
                else:
                    tgt_mask = model.get_tgt_mask(vitals.to(device).shape[-1]).to(device)
                    sofap_t = model(vitals.to(device), tgt_mask, key_mask.to(device))
                    # if seq_handel = 'scale', sofa_p
                # if seq_handel = 'scale':
                # else:
                if args.loss_rule == 'mean':
                    pred = torch.stack([sofap_t[i][key_mask[i] == 0].mean(dim=-2) for i in range(len(sofap_t))])
                elif args.loss_rule == 'last':
                    pred = torch.stack([sofap_t[i][key_mask[i] == 0][-1, :] for i in range(len(sofap_t))])
                loss_v = ce_val_loss(pred, sofa_t.squeeze(-1))

                y_list.append(sofa_t)
                y_pred_list.append(pred)
                loss_val.append(loss_v)

        loss_te = np.mean(torch.stack(loss_val, dim=0).cpu().detach().numpy())
        val_acc = cal_acc(y_pred_list, y_list)
        # calculate AUROC
        y_l = torch.concat(y_list).cpu().numpy()
        y_pred_l = np.concatenate([f_sm(y_pred_list[i]).cpu().numpy() for i in range(len(y_pred_list))])
        dev_roc = roc_auc_score(y_l.squeeze(-1), y_pred_l[:, 1])

        if args.cal_pos_acc == True:
            val_pos_acc = cal_pos_acc(y_pred_list, y_list, pos_ind=1)
            print('Validation pos acc is %.2f%%' % (val_pos_acc * 100))

        print('Validation acc is %.2f%%, validation ROC is %.2f' % (val_acc * 100, dev_roc))

        if j == 0:
            best_auroc = dev_roc
            best_model = copy.deepcopy(model)
        if dev_roc > best_auroc:
            best_auroc = dev_roc
            best_model = copy.deepcopy(model)
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                print('Start next fold')
                break
        print('Epoch %d, : Train loss is %.4f, validation loss is %.4f' % (j, loss_avg, loss_te))

    return best_model


def plot_confusion_matrix(y_list, y_pred_list, title='Confusion matrix', label_x=None, label_y=None):
    '''

    :param y_list: a list of tensor, each tensor is a batch of labels
    :param y_pred_list: a list of tensor, each tensor is a batch of predictions
    :param title: title of the plot
    :param label_x: label of x axis
    :param label_y: label of y axis
    :return: figure of confusion matrix
    '''
    num_class = y_pred_list[0].shape[-1]
    y_label = torch.concat(y_list).detach().cpu().numpy()
    pred_t = torch.concat(y_pred_list)
    y_pred = torch.argmax(pred_t, dim=-1).unsqueeze(-1).detach().cpu().numpy()

    cm = metrics.confusion_matrix(y_label, y_pred)
    cf_matrix = cm / np.repeat(np.expand_dims(np.sum(cm, axis=1), axis=-1), num_class, axis=1)
    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    # percentage based on true label
    gr = (cm / np.repeat(np.expand_dims(np.sum(cm, axis=1), axis=-1), num_class, axis=1)).flatten()
    group_percentages = ['{0:.2%}'.format(value) for value in gr]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_percentages, group_counts)]

    labels = np.asarray(labels).reshape(num_class, num_class)

    if label_x is not None:
        xlabel = label_x
        ylabel = label_y
    else:
        xlabel = ['Pred-%d' % i for i in range(num_class)]
        ylabel = ['%d' % i for i in range(num_class)]

    sns.set(font_scale=1.5)

    hm = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='OrRd', \
                     annot_kws={"fontsize": 16}, xticklabels=xlabel, yticklabels=ylabel, cbar=False)
    hm.set(title=title)
    fig = plt.gcf()
    plt.show()
    return fig


def plot_confusion_matrix_cpu(y_list, y_pred_list, title='Confusion matrix', label_x=None, label_y=None):
    # cpu version
    num_class = y_pred_list[0].shape[-1]
    y_pred = np.argmax(y_pred_list, axis=-1)

    cm = metrics.confusion_matrix(y_list, y_pred)
    cf_matrix = cm / np.repeat(np.expand_dims(np.sum(cm, axis=1), axis=-1), num_class, axis=1)
    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    # percentage based on true label
    gr = (cm / np.repeat(np.expand_dims(np.sum(cm, axis=1), axis=-1), num_class, axis=1)).flatten()
    group_percentages = ['{0:.2%}'.format(value) for value in gr]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_percentages, group_counts)]

    labels = np.asarray(labels).reshape(num_class, num_class)

    if label_x is not None:
        xlabel = label_x
        ylabel = label_y
    else:
        xlabel = ['Pred-%d' % i for i in range(num_class)]
        ylabel = ['%d' % i for i in range(num_class)]

    sns.set(font_scale=1.5)

    hm = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='OrRd', \
                     annot_kws={"fontsize": 16}, xticklabels=xlabel, yticklabels=ylabel, cbar=False)
    hm.set(title=title)
    fig = plt.gcf()
    plt.show()
    return fig


# plot auprc and roc curve
# from get_evalacc_results
def plot_auprc(y_list, y_pred_list):
    '''
    :param y_list: a list of tensor, each tensor is a batch of labels
    :param y_pred_list: a list of tensor, each tensor is a batch of predictions
    :return: figure of auprc
    '''
    binary_label = torch.concat(y_list).detach().cpu().numpy()
    binary_outputs = softmax(torch.concat(y_pred_list)).detach().cpu().numpy()
    metrics.PrecisionRecallDisplay.from_predictions(binary_label, binary_outputs[:, 1])

    no_skill = len(binary_label[binary_label == 1]) / len(binary_label)
    # plot the no skill precision-recall curve
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill (AP = %.2f)' % no_skill)
    plt.legend()
    fig = plt.gcf()
    plt.show()
    return fig


def plot_roc(y_list, y_pred_list):
    '''
    :param y_list: a list of tensor, each tensor is a batch of labels
    :param y_pred_list: a list of tensor, each tensor is a batch of predictions
    :return: figure of roc
    '''
    binary_label = torch.concat(y_list).detach().cpu().numpy()
    binary_outputs = softmax(torch.concat(y_pred_list)).detach().cpu().numpy()
    metrics.RocCurveDisplay.from_predictions(binary_label, binary_outputs[:, 1])

    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    # no_skill = len(binary_label[binary_label==1]) / len(binary_label)
    # plot the no skill precision-recall curve
    # plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill (AP = %.2f)'%no_skill)
    plt.legend()
    plt.xlabel(xlabel='FPR', fontsize=8)
    plt.ylabel(ylabel='TPR', fontsize=8)
    fig = plt.gcf()
    plt.show()
    return fig


# calculate acc
def cal_acc(pred, label):
    '''
    :param pred: a list of tensor, each tensor is a batch of predictions
    :param label: a list of tensor, each tensor is a batch of labels
    :return: accuarcy
    '''
    pred_t = torch.concat(pred)
    prediction = torch.argmax(pred_t, dim=-1).unsqueeze(-1)
    label_t = torch.concat(label)
    acc = (prediction == label_t).sum() / len(pred_t)
    return acc


# for classification get evaluation results
def get_evalacc_results(args, model, test_loader):
    '''
    :param args: arguments
    :param model: model to be evaluated
    :param test_loader: test data loader
    :return: y_list: a list of tensor, each tensor is a batch of labels
                y_pred_list: a list of tensor, each tensor is a batch of predictions
                td_list: a list of tensor, each tensor is a batch of input data
                loss_te: test loss
                val_acc: test accuracy
    '''
    model.eval()
    y_list = []
    y_pred_list = []
    td_list = []
    loss_val = []
    with torch.no_grad():  # validation does not require gradient

        for vitals, target, key_mask in test_loader:
            # ti_test = Variable(torch.FloatTensor(ti)).to(device)
            td_test = Variable(vitals.float().to(device))
            sofa_t = Variable(target.long().to(device))

            # tgt_mask_test = model.get_tgt_mask(td_test.shape[-1]).to(device)
            if args.model_name == 'TCN':

                sofap_t = model(td_test)
            elif args.model_name == 'RNN':
                # x_lengths have to be a 1d tensor
                td_transpose = td_test.transpose(1, 2)
                x_lengths = torch.LongTensor([len(key_mask[i][key_mask[i] == 0]) for i in range(key_mask.shape[0])])
                sofap_t = model(td_transpose, x_lengths)
            else:
                tgt_mask = model.get_tgt_mask(vitals.to(device).shape[-1]).to(device)
                sofap_t = model(vitals.to(device), tgt_mask, key_mask.to(device))
                # if seq_handel = 'scale', sofa_p
            # if seq_handel = 'scale':
            # else:
            if args.loss_rule == 'mean':
                pred = torch.stack([sofap_t[i][key_mask[i] == 0].mean(dim=-2) for i in range(len(sofap_t))])
            elif args.loss_rule == 'last':
                pred = torch.stack([sofap_t[i][key_mask[i] == 0][-1, :] for i in range(len(sofap_t))])

            loss_v = ce_loss(pred, sofa_t.squeeze(-1))

            y_list.append(sofa_t)
            y_pred_list.append(pred)
            loss_val.append(loss_v)
            td_list.append(td_test)

        loss_te = np.mean(torch.stack(loss_val, dim=0).cpu().detach().numpy())
        val_acc = cal_acc(y_pred_list, y_list)

    return y_list, y_pred_list, td_list, loss_te, val_acc


def get_eval_results(model, test_loader):
    model.eval()
    y_list = []
    y_pred_list = []
    td_list = []
    loss_val = []
    with torch.no_grad():  # validation does not require gradient

        for vitals, target, key_mask in test_loader:
            # ti_test = Variable(torch.FloatTensor(ti)).to(device)
            td_test = Variable(torch.FloatTensor(vitals)).to(device)
            sofa_t = Variable(torch.FloatTensor(target)).to(device)

            tgt_mask_test = model.get_tgt_mask(td_test.shape[-1]).to(device)
            sofap_t = model(td_test, tgt_mask_test, key_mask.to(device))

            loss_v = loss_fn.mse_maskloss(sofap_t, sofa_t, key_mask.to(device))
            y_list.append(sofa_t.cpu().detach().numpy())
            y_pred_list.append(sofap_t.cpu().detach().numpy())
            loss_val.append(loss_v)
    loss_te = np.mean(torch.stack(loss_val, dim=0).cpu().detach().numpy())

    return y_list, y_pred_list, td_list, loss_te




