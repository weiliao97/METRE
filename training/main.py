# parameter tuning
import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import os
from tqdm import tqdm
import importlib
import models
import prepare_data
import make_optimizer
import utils
import loss_fn
importlib.reload(models)
importlib.reload(make_optimizer)
importlib.reload(prepare_data)
importlib.reload(utils)
importlib.reload(loss_fn)
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import scipy.stats as st
from datetime import date
today = date.today()
date = today.strftime("%m%d")
kf = KFold(n_splits=10, random_state=42, shuffle=True)
f_sm = nn.Softmax(dim=1)

# count model trainable params
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# make dir and write json files
def write_json(target_path, target_file, data):
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)
        except Exception as e:
            print(e)
            raise
    with open(os.path.join(target_path, target_file), 'w') as f:
        json.dump(data, f)

# combine train and val and based on the cross validation index, redistribute train and val
def get_cv_data(train_data, dev_data, train_target, dev_target, train_index, dev_index):
    trainval_head = train_data + dev_data
    trainval_static = np.concatenate((train_target, dev_target), axis=0)
    train_cv = [trainval_head[i] for i in train_index]
    train_cvl = [trainval_static[i] for i in train_index]
    dev_cv = [trainval_head[i] for i in dev_index]
    dev_cvl = [trainval_static[i] for i in dev_index]
    return train_cv, dev_cv, np.asarray(train_cvl), np.asarray(dev_cvl)

# calcuate accuracy
def cal_acc(pred, label):
    pred_t = torch.concat(pred)
    prediction = torch.argmax(pred_t, dim=-1).unsqueeze(-1)
    label_t = torch.concat(label)
    acc = (prediction == label_t).sum() / len(pred_t)
    return acc

# calcualte accuracy for positive classes
def cal_pos_acc(pred, label, pos_ind):
    pred_t = torch.concat(pred)
    prediction = torch.argmax(pred_t, dim=-1).unsqueeze(-1)
    label_t = torch.concat(label)
    # positive index
    ind = [i for i in range(len(pred_t)) if label_t[i] == pos_ind]
    acc = (prediction[ind] == label_t[ind]).sum() / len(ind)
    return acc

# filter ICU stays based on LOS needed: 48+6, 4+6, 12+6
def filter_los(static_data, vitals_data, thresh, gap):
    # (200, 80)
    los = [i.shape[1] for i in vitals_data]
    ind = [i for i in range(len(los)) if los[i] >= (thresh + gap) and np.isnan(static_data[i, 0]) == False]
    vitals_reduce = [vitals_data[i][:, :thresh] for i in ind]
    static_data = static_data[ind]
    return static_data, vitals_reduce

# filter for the ARF task
def filter_arf(args, vital):
    vital_reduce = []
    target = []
    for i in range(len(vital)):
        arf_flag = np.where(vital[i][184, :] == 1)[0]
        peep_flag = np.union1d(np.where(vital[i][157, :] == 1)[0], np.where(vital[i][159, :] == 1)[0])
        if len(arf_flag) == 0:
            if len(peep_flag) > 0:
                if peep_flag[0] >= (args.thresh + args.gap):
                    vital_reduce.append(vital[i][:, :args.thresh])
                    target.append(1)
            else:
                vital_reduce.append(vital[i][:, :args.thresh])
                target.append(0)
        elif len(arf_flag) > 0:
            if arf_flag[0] >= (args.thresh + args.gap):
                if (len(peep_flag) > 0 and peep_flag[0] >= (args.thresh + args.gap)) or len(peep_flag) == 0:
                    vital_reduce.append(vital[i][:, :args.thresh])
                    target.append(1)
    return vital_reduce, np.asarray(target)

# filter for the shock prediction task
def filter_shock(args, vital):
    vital_reduce = []
    target = []
    for i in range(len(vital)):
        shock_flag = np.where(vital[i][186:191].sum(axis=0) >= 1)[0]
        if len(shock_flag) == 0:
            vital_reduce.append(vital[i][:, :args.thresh])
            target.append(0)
        elif len(shock_flag) > 0:
            if shock_flag[0] >= (args.thresh + args.gap):
                vital_reduce.append(vital[i][:, :args.thresh])
                target.append(1)
    return vital_reduce, np.asarray(target)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for Tranformer models")

    parser.add_argument("--dataset_path", type=str, help="path to the dataset")
    parser.add_argument("--dataset_path_cv", type=str, help="path to the dataset")
    parser.add_argument("--model_name", type=str, default='TCN', choices=['Trans', 'TCN', 'RNN'])
    parser.add_argument("--rnn_type", type=str, default='lstm', choices=['rnn', 'lstm', 'gru'])

    # important, which target to use as the prediction taregt 0: hospital mortality, 1: ARF, 2: shock
    parser.add_argument("--target_index", type=int, default=0, help="Which static column to target")
    parser.add_argument("--output_classes", type=int, default=2, help="Which static column to target")
    parser.add_argument("--cal_pos_acc", action='store_false', default=True,
                        help="Whethe calculate the acc of the positive class")
    parser.add_argument("--filter_los", action='store_false', default=True,
                        help="Whether filter the first xxx hours of stay")
    parser.add_argument("--thresh", type=int, default=48, help="how many hours of data to use")
    parser.add_argument("--gap", type=int, default=6, help="gap hours between record stop and data used in training")

    # model parameters
    # TCN
    parser.add_argument("--kernel_size", type=int, default=3, help="Dimension of the model")
    parser.add_argument("--dropout", type=float, default=0.2, help="Model dropout")
    parser.add_argument("--reluslope", type=float, default=0.1, help="Relu slope in the fc model")
    parser.add_argument('--num_channels', nargs='+', help='<Required> Set flag')
    # LSTM
    parser.add_argument("--hidden_dim", type=int, default=512, help="RNN hidden dim")
    parser.add_argument("--layer_dim", type=int, default=3, help="RNN layer dim")
    parser.add_argument("--idrop", type=float, default=0, help="RNN drop out in the very beginning")

    # transformer
    parser.add_argument("--d_model", type=int, default=256, help="Dimension of the model")
    parser.add_argument("--n_head", type=int, default=8, help="Attention head of the model")
    parser.add_argument("--dim_ff_mul", type=int, default=4, help="Dimension of the feedforward model")
    parser.add_argument("--num_enc_layer", type=int, default=2, help="Number of encoding layers")

    # learning parameters
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--data_batching", type=str, default='same', choices=['same', 'close', 'random'],
                        help='How to batch data')
    parser.add_argument("--bs", type=int, default=16, help="Batch size for training")
    # learning rate
    parser.add_argument('--warmup', action='store_true', default = False, help="whether use learning rate warm up")
    parser.add_argument('--lr_factor', type=int, default=0.1, help="warmup_learning rate factor")
    parser.add_argument('--lr_steps', type=int, default=2000, help="warmup_learning warm up steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")  # could be overwritten by warm up
    # loss compute, mean or last , output (16, 24, 2) for RNN and TCN
    parser.add_argument("--loss_rule", type=str, default='last', choices=['mean', 'last'])

    # Parse and return arguments
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_map = {0: 'hosp_mort', 1: 'ARF', 2: 'shock'}
    # load data
    data_label = np.load(args.dataset_path, allow_pickle=True).item()
    train_head = data_label['train_head']
    static_train_filter = data_label['static_train_filter']
    dev_head = data_label['dev_head']
    static_dev_filter = data_label['static_dev_filter']
    test_head = data_label['test_head']
    static_test_filter = data_label['static_test_filter']
    s_train = np.stack(static_train_filter, axis=0)
    s_dev = np.stack(static_dev_filter, axis=0)
    s_test = np.stack(static_test_filter, axis=0)
    # load cross validation data from the other database
    data_label = np.load(args.dataset_path_cv, allow_pickle=True).item()
    etrain_head = data_label['train_head']
    estatic_train_filter = data_label['static_train_filter']
    edev_head = data_label['dev_head']
    estatic_dev_filter = data_label['static_dev_filter']
    etest_head = data_label['test_head']
    estatic_test_filter = data_label['static_test_filter']
    es_train = np.stack(estatic_train_filter, axis=0)
    es_dev = np.stack(estatic_dev_filter, axis=0)
    es_test = np.stack(estatic_test_filter, axis=0)

    print('Running target %d, thresh %d, gap %d, model %s' % (args.target_index, args.thresh, args.gap, args.model_name))
    workname = date + '_%s' % task_map[args.target_index] + '_%dh' % args.thresh + '_%sh' % args.gap + '_%s' % (args.model_name.lower())
    print(workname)
    args.checkpoint_model = workname
    if args.target_index == 0 and args.filter_los:
        print('Before filtering, train size is %d' % (len(train_head)))
        train_label, train_data = filter_los(s_train, train_head, args.thresh, args.gap)
        dev_label, dev_data = filter_los(s_dev, dev_head, args.thresh, args.gap)
        test_label, test_data = filter_los(s_test, test_head, args.thresh, args.gap)
        print('After filtering, train size is %d' % (len(train_data)))
        train_label = train_label[:, 0]
        dev_label = dev_label[:, 0]
        test_label = test_label[:, 0]

    if args.target_index == 1:  # arf
        print('Before filtering, train size is %d' % (len(train_head)))
        train_data, train_label = filter_arf(args, train_head)
        dev_data, dev_label = filter_arf(args, dev_head)
        test_data, test_label = filter_arf(args, test_head)
        print('After filtering, train size is %d' % (len(train_data)))

    if args.target_index == 2:  # shock
        print('Before filtering, train size is %d' % (len(train_head)))
        train_data, train_label = filter_shock(args, train_head)
        dev_data, dev_label = filter_shock(args, dev_head)
        test_data, test_label = filter_shock(args, test_head)
        print('After filtering, train size is %d' % (len(train_data)))

    trainval_data = train_data + dev_data

    # for cross validation
    if args.target_index == 0 and args.filter_los:
        print('Before filtering, train size is %d' % (len(etrain_head)))
        es_train1, etrain_data = filter_los(es_train, etrain_head, args.thresh, args.gap)
        es_dev1, edev_data = filter_los(es_dev, edev_head, args.thresh, args.gap)
        es_test1, etest_data = filter_los(es_test, etest_head, args.thresh, args.gap)
        print('After filtering, train size is %d' % (len(etrain_data)))
        etrain_label = es_train1[:, 0]
        edev_label = es_dev1[:, 0]
        etest_label = es_test1[:, 0]

    elif args.target_index == 1:
        etrain_data, etrain_label = filter_arf(args, etrain_head)
        edev_data, edev_label = filter_arf(args, edev_head)
        etest_data, etest_label = filter_arf(args, etest_head)

    elif args.target_index == 2:
        etrain_data, etrain_label = filter_shock(args, etrain_head)
        edev_data, edev_label = filter_shock(args, edev_head)
        etest_data, etest_label = filter_shock(args, etest_head)

    crossval_head = etrain_data + edev_data + etest_data
    crossval_target = np.concatenate((etrain_label, edev_label, etest_label), axis=0)

    # result_dict to log and save data
    result_dict = {}
    # create model
    if args.model_name == 'TCN':
        print('Creating TCN')
        model = models.TemporalConv(num_inputs=200, num_channels=[int(i) for i in args.num_channels], \
                                    kernel_size=args.kernel_size, dropout=args.dropout, \
                                    output_class=args.output_classes)
        torch.save(model.state_dict(), '/content/start_weights.pt')
        print('Saving Initial Weights')
        print("Trainable params in TCN is %d" % count_parameters(model))
    elif args.model_name == 'RNN':
        model = models.RecurrentModel(cell=args.rnn_type, hidden_dim=args.hidden_dim,
                                      layer_dim=args.layer_dim, \
                                      output_dim=args.output_classes, dropout_prob=args.dropout,
                                      idrop=args.idrop)

        torch.save(model.state_dict(), '/content/start_weights.pt')
        print('Saving Initial Weights')
        print("Trainable params in RNN is %d" % count_parameters(model))

    else:
        model = models.Trans_encoder(feature_dim=200, d_model=args.d_model, \
                                     nhead=args.n_head, d_hid=args.dim_ff_mul * args.d_model, \
                                     nlayers=args.num_enc_layer, out_dim=args.output_classes, dropout=args.dropout)
        torch.save(model.state_dict(), '/content/start_weights.pt')
        print('Saving Initial Weights')
        print("Trainable params in RNN is %d" % count_parameters(model))

    model.to(device)
    best_loss = 1e4
    best_acc = 0.5
    best_diff = 0.1
    best_roc = 0.5

    # loss fn and optimizer
    ce_loss = torch.nn.CrossEntropyLoss()
    if args.warmup == True:
        print('Using warm up')
        model_opt = make_optimizer.NoamOpt(args.d_model, args.lr_factor, args.lr_steps,
                                           torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98),
                                                            eps=1e-9))
    else:
        print('No warm up')
        model_opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        torch.save(model_opt.state_dict(), '/content/start_weights_opt.pt')

    for c_fold, (train_index, test_index) in enumerate(kf.split(trainval_data)):
        best_loss = 1e4
        patience = 0
        if c_fold >= 1:
            model.load_state_dict(torch.load('/content/start_weights.pt'))
            model_opt.load_state_dict(torch.load('/content/start_weights_opt.pt'))
        print('Starting Fold %d' % c_fold)
        print("TRAIN:", len(train_index), "TEST:", len(test_index))

        train_cv, dev_cv, train_labelcv, dev_labelcv = get_cv_data(train_data, dev_data, train_label, dev_label, train_index, test_index)
        print('Compiled another CV data')
        train_dataloader, dev_dataloader, test_dataloader = prepare_data.get_data_loader( \
            args, train_cv, dev_cv, test_data, train_labelcv, dev_labelcv, test_label)

        ctype, count = np.unique(dev_labelcv, return_counts=True)
        total_dev_samples = len(dev_labelcv)
        weights_per_class = torch.FloatTensor([total_dev_samples / k / len(ctype) for k in count]).to(
            device)
        ce_val_loss = nn.CrossEntropyLoss(weight=weights_per_class)

        best_model = utils.train_model(args, c_fold, model, model_opt, train_dataloader,
                                       dev_dataloader, ce_loss, ce_val_loss)

        # test auroc on test set
        y_list, y_pred_list, td_list, loss_te, val_acc = utils.get_evalacc_results(args, best_model, test_dataloader)
        y_l = torch.concat(y_list).cpu().numpy()
        y_pred_l = np.concatenate([f_sm(y_pred_list[i]).cpu().numpy() for i in range(len(y_pred_list))])
        test_roc = roc_auc_score(y_l.squeeze(-1), y_pred_l[:, 1])

        if test_roc > best_roc:
            best_roc = test_roc
            print('Save a model with best roc %.3f' % best_roc)
            torch.save(best_model.state_dict(),
                       './checkpoints/' + args.checkpoint_model + '_fold%d' %c_fold + '_best_roc_%.3f.pt' % best_roc)
            # for best roc on test set models, perform bootstrapping on both test set and the whole another set
            # save the results in a dictionary and save that dictionary regularly
            roc = []
            prc = []
            for i in tqdm(range(1000)):
                test_index = np.random.choice(len(test_label), 1000)
                test_i = [test_data[i] for i in test_index]
                test_t = test_label[test_index]
                test_dataloader = prepare_data.get_test_loader(args, test_i, test_t)
                # test auroc on test set
                y_list, y_pred_list, td_list, loss_te, val_acc = utils.get_evalacc_results(args, best_model,
                                                                                           test_dataloader)
                y_l = torch.concat(y_list).cpu().numpy()
                y_pred_l = np.concatenate(
                    [f_sm(y_pred_list[i]).cpu().numpy() for i in range(len(y_pred_list))])
                # tpr, tnr = get_tp_tn(y_l.squeeze(-1), y_pred_l[:, 1])
                test_roc = roc_auc_score(y_l.squeeze(-1), y_pred_l[:, 1])
                test_prc = average_precision_score(y_l.squeeze(-1), y_pred_l[:, 1])
                roc.append(test_roc)
                prc.append(test_prc)
            # create 95% confidence interval for population mean weight
            result_dict['fold%d'%c_fold] = ['%.3f' % np.mean(roc)]
            result_dict['fold%d'%c_fold].append(
                '(%.3f-%.3f)' % st.t.interval(alpha=0.95, df=len(roc), loc=np.mean(roc), scale=np.std(roc)))
            result_dict['fold%d'%c_fold].append('%.3f' % np.mean(prc))
            result_dict['fold%d'%c_fold].append(
                '(%.3f-%.3f)' % st.t.interval(alpha=0.95, df=len(prc), loc=np.mean(prc), scale=np.std(prc)))
            result_dict['fold%d'%c_fold].append(len(test_label))

            roc = []
            prc = []
            for i in tqdm(range(1000)):
                test_index = np.random.choice(len(crossval_target), 1000)
                test_i = [crossval_head[i] for i in test_index]
                test_t = crossval_target[test_index]
                test_dataloader = prepare_data.get_test_loader(args, test_i, test_t)
                # test auroc on test set
                y_list, y_pred_list, td_list, loss_te, val_acc = utils.get_evalacc_results(args, best_model,
                                                                                           test_dataloader)
                y_l = torch.concat(y_list).cpu().numpy()
                y_pred_l = np.concatenate(
                    [f_sm(y_pred_list[i]).cpu().numpy() for i in range(len(y_pred_list))])
                test_roc = roc_auc_score(y_l.squeeze(-1), y_pred_l[:, 1])
                test_prc = average_precision_score(y_l.squeeze(-1), y_pred_l[:, 1])
                roc.append(test_roc)
                prc.append(test_prc)
            # create 95% confidence interval for population mean weight
            result_dict['fold%d'%c_fold].append('%.3f' % np.mean(roc))
            result_dict['fold%d'%c_fold].append(
                '(%.3f-%.3f)' % st.t.interval(alpha=0.95, df=len(roc), loc=np.mean(roc), scale=np.std(roc)))
            result_dict['fold%d'%c_fold].append('%.3f' % np.mean(prc))
            result_dict['fold%d'%c_fold].append(
                '(%.3f-%.3f)' % st.t.interval(alpha=0.95, df=len(prc), loc=np.mean(prc), scale=np.std(prc)))
            result_dict['fold%d'%c_fold].append(len(crossval_target))

    write_json('./checkpoints', args.checkpoint_model + '.json', result_dict)