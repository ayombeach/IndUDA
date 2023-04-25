import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

from solver import utils as solver_utils
from utils.utils import to_cuda, to_onehot, to_norm
from torch import optim, squeeze, unsqueeze
from solver import clustering
from discrepancy.cdd import CDD
from math import ceil as ceil
from solver.base_solver import BaseSolver
from model import Inv_Model
from copy import deepcopy
from model.Inv_Model import InvNet, subnet_constructor
from torch.autograd import Variable

class INNSolver(BaseSolver):
    def __init__(self, net, dataloader, bn_domain_map={}, resume=None, **kwargs):
        super(INNSolver, self).__init__(net, dataloader,
                                        bn_domain_map, resume=resume, **kwargs)

        if len(self.bn_domain_map) == 0:
            self.bn_domain_map = {self.source_name: 0, self.target_name: 1}

        self.clustering_source_name = 'clustering_' + self.source_name
        self.clustering_target_name = 'clustering_' + self.target_name
        assert ('categorical' in self.train_data)

        # num_layers = len(self.net.module.FC) + 1
        self.num_layers = 1
        self.cdd = CDD(kernel_num=self.opt.CDD.KERNEL_NUM, kernel_mul=self.opt.CDD.KERNEL_MUL,
                       num_layers=self.num_layers, num_classes=self.opt.DATASET.NUM_CLASSES,
                       intra_only=self.opt.CDD.INTRA_ONLY)

        self.discrepancy_key = 'intra' if self.opt.CDD.INTRA_ONLY else 'cdd'
        self.clustering = clustering.Clustering(self.opt.CLUSTERING.EPS,
                                                self.opt.CLUSTERING.FEAT_KEY,
                                                self.opt.CLUSTERING.BUDGET)
        self.clustered_target_samples = {}
        # self.mask = torch.cat((torch.ones(int(self.opt.DATASET.NUM_CLASSES), 1024, 1, 1), torch.zeros(int(self.opt.DATASET.NUM_CLASSES), 1024, 1, 1)), 1).cuda()
        self.mask = torch.zeros(int(self.opt.DATASET.NUM_CLASSES), 2048, 1, 1)
        self.index = torch.zeros(int(self.opt.DATASET.NUM_CLASSES)).cuda()

    def complete_training(self):
        if self.loop >= self.opt.TRAIN.MAX_LOOP:
            return True

        if 'target_centers' not in self.history or \
                'ts_center_dist' not in self.history or \
                'target_labels' not in self.history:
            return False

        if len(self.history['target_centers']) < 2 or \
                len(self.history['ts_center_dist']) < 1 or \
                len(self.history['target_labels']) < 2:
            return False

        # target centers along training
        target_centers = self.history['target_centers']
        eval1 = torch.mean(self.clustering.Dist.get_dist(target_centers[-1],
                                                         target_centers[-2])).item()

        # target-source center distances along training
        eval2 = self.history['ts_center_dist'][-1].item()

        # target labels along training
        path2label_hist = self.history['target_labels']
        paths = self.clustered_target_samples['data']
        num = 0
        for path in paths:
            pre_label = path2label_hist[-2][path]
            cur_label = path2label_hist[-1][path]
            if pre_label != cur_label:
                num += 1
        eval3 = 1.0 * num / len(paths)

        return (eval1 < self.opt.TRAIN.STOP_THRESHOLDS[0] and \
                eval2 < self.opt.TRAIN.STOP_THRESHOLDS[1] and \
                eval3 < self.opt.TRAIN.STOP_THRESHOLDS[2])

    def solve(self):
        stop = False
        if self.resume:
            self.iters += 1
            self.loop += 1

        while True:
            # updating the target label hypothesis through clustering
            target_hypt = {}
            filtered_classes = []
            with torch.no_grad():
                # self.update_ss_alignment_loss_weight()
                print('Clustering based on %s...' % self.source_name)
                self.update_labels()
                self.clustered_target_samples = self.clustering.samples
                target_centers = self.clustering.centers
                center_change = self.clustering.center_change
                path2label = self.clustering.path2label

                # updating the history
                self.register_history('target_centers', target_centers,
                                      self.opt.CLUSTERING.HISTORY_LEN)
                self.register_history('ts_center_dist', center_change,
                                      self.opt.CLUSTERING.HISTORY_LEN)
                self.register_history('target_labels', path2label,
                                      self.opt.CLUSTERING.HISTORY_LEN)

                if self.clustered_target_samples is not None and \
                        self.clustered_target_samples['gt'] is not None:
                    preds = to_onehot(self.clustered_target_samples['label'],
                                      self.opt.DATASET.NUM_CLASSES)
                    gts = self.clustered_target_samples['gt']
                    res = self.model_eval(preds, gts)
                    print('Clustering %s: %.4f' % (self.opt.EVAL_METRIC, res))

                # check if meet the stop condition
                stop = self.complete_training()
                if stop: break

                # filtering the clustering results
                target_hypt, filtered_classes = self.filtering()

                # update dataloaders
                self.construct_categorical_dataloader(target_hypt, filtered_classes)
                # update train data setting
                self.compute_iters_per_loop(filtered_classes)

            # k-step update of network parameters through forward-backward process
            # self.get_class_mask()
            self.update_network(filtered_classes)

            self.loop += 1

        print('Training Done!')

    def update_labels(self):
        net = self.net

        net.eval()
        opt = self.opt

        source_dataloader = self.train_data[self.clustering_source_name]['loader']
        net.module.set_bn_domain(self.bn_domain_map[self.source_name])

        source_centers = solver_utils.get_centers(net.module,
                                                  source_dataloader, self.opt.DATASET.NUM_CLASSES,
                                                  self.opt.CLUSTERING.FEAT_KEY)
        init_target_centers = source_centers

        target_dataloader = self.train_data[self.clustering_target_name]['loader']
        net.module.set_bn_domain(self.bn_domain_map[self.target_name])

        self.clustering.set_init_centers(init_target_centers)
        self.clustering.feature_clustering(net, target_dataloader)

    def filtering(self):
        threshold = self.opt.CLUSTERING.FILTERING_THRESHOLD
        min_sn_cls = self.opt.TRAIN.MIN_SN_PER_CLASS
        target_samples = self.clustered_target_samples

        # filtering the samples
        chosen_samples = solver_utils.filter_samples(
            target_samples, threshold=threshold)

        # filtering the classes
        filtered_classes = solver_utils.filter_class(
            chosen_samples['label'], min_sn_cls, self.opt.DATASET.NUM_CLASSES)

        print('The number of filtered classes: %d.' % len(filtered_classes))
        return chosen_samples, filtered_classes

    def construct_categorical_dataloader(self, samples, filtered_classes):
        # update self.dataloader
        target_classwise = solver_utils.split_samples_classwise(
            samples, self.opt.DATASET.NUM_CLASSES)

        dataloader = self.train_data['categorical']['loader']
        classnames = dataloader.classnames
        dataloader.class_set = [classnames[c] for c in filtered_classes]
        dataloader.target_paths = {classnames[c]: target_classwise[c]['data'] \
                                   for c in filtered_classes}
        dataloader.num_selected_classes = min(self.opt.TRAIN.NUM_SELECTED_CLASSES, len(filtered_classes))
        dataloader.construct()

    def CAS(self):
        samples = self.get_samples('categorical')

        source_samples = samples['Img_source']
        source_sample_paths = samples['Path_source']
        source_nums = [len(paths) for paths in source_sample_paths]

        target_samples = samples['Img_target']
        target_sample_paths = samples['Path_target']
        target_nums = [len(paths) for paths in target_sample_paths]

        source_sample_labels = samples['Label_source']
        self.selected_classes = [labels[0].item() for labels in source_sample_labels]

        gt = torch.zeros(self.opt.TRAIN.SOURCE_CLASS_BATCH_SIZE * len(source_nums))
        for i in range(len(source_sample_labels)):
            gt[i * self.opt.TRAIN.SOURCE_CLASS_BATCH_SIZE:(i + 1) * self.opt.TRAIN.SOURCE_CLASS_BATCH_SIZE] = \
                source_sample_labels[i]
        assert (self.selected_classes ==
                [labels[0].item() for labels in samples['Label_target']])
        return source_samples, source_nums, target_samples, target_nums, gt

    def prepare_feats(self, feats):
        return [feats[key] for key in feats if key in self.opt.CDD.ALIGNMENT_FEAT_KEYS]

    def compute_iters_per_loop(self, filtered_classes):
        self.iters_per_loop = int(
            len(self.train_data['categorical']['loader'])) * self.opt.TRAIN.UPDATE_EPOCH_PERCENTAGE
        print('Iterations in one loop: %d' % (self.iters_per_loop))

    def update_network(self, filtered_classes):
        # initial configuration
        stop = False
        update_iters = 0

        self.train_data[self.source_name]['iterator'] = \
            iter(self.train_data[self.source_name]['loader'])
        self.train_data[self.target_name]['iterator'] = \
            iter(self.train_data[self.target_name]['loader'])
        self.train_data['categorical']['iterator'] = \
            iter(self.train_data['categorical']['loader'])

        while not stop:
            # update learning rate
            self.update_lr()

            # set the status of network
            self.net.train()
            # self.net.zero_grad()

            loss = 0
            swap_loss = 0
            align_loss = 0
            # coventional sampling for training on labeled source data
            source_sample = self.get_samples(self.source_name)
            source_data, source_gt = source_sample['Img'], \
                                     source_sample['Label']

            source_data = to_cuda(source_data)
            source_gt = to_cuda(source_gt)
            # train the source model
            self.net.module.set_bn_domain(self.bn_domain_map[self.source_name])
            source_feature = self.F(source_data)
            source_logits = self.C(source_feature.squeeze())

            self.get_mask(source_feature, source_gt)
            ce_loss = self.CELoss(source_logits, source_gt)

            ce_loss.backward()
            ce_loss_iter += ce_loss
            loss += ce_loss
            if len(filtered_classes) > 0:
                # update the network parameters
                # 1) class-aware sampling
                source_samples_cls, source_nums_cls, \
                target_samples_cls, target_nums_cls, gt = self.CAS()
                # 2) forward and compute the loss
                source_cls_concat = torch.cat([to_cuda(samples)
                                               for samples in source_samples_cls], dim=0)
                target_cls_concat = torch.cat([to_cuda(samples)
                                               for samples in target_samples_cls], dim=0)

                self.net.module.set_bn_domain(self.bn_domain_map[self.source_name])
                feats_source = self.F(source_cls_concat)
                feats_s = feats_source.clone().detach()
                probs_source = nn.Softmax(dim=1)(self.C(feats_source.squeeze()))
                probs_s = nn.Softmax(dim=1)(self.C(feats_s.squeeze()))

                self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
                feats_target = self.F(target_cls_concat)
                feats_t = feats_target.clone().detach()
                probs_target = nn.Softmax(dim=1)(self.C(feats_target.squeeze()))
                probs_t = nn.Softmax(dim=1)(self.C(feats_t.squeeze()))

                feats_source_inn = self.INN(feats_s)
                feats_target_inn = self.INN(feats_t)

                mask = self.get_gt_mask(self.mask, gt)# .bool()
                # torch.save(self.mask, 'mask.npy')
                # mask = self.get_mask(feats_source, gt)
                feats_source_class = torch.masked_select(feats_source_inn, mask.bool()).view(len(gt), -1)
                feats_target_class = torch.masked_select(feats_target_inn, mask.bool()).view(len(gt), -1)

                feats_source_clas = feats_source_inn * mask
                feats_target_clas = feats_target_inn * mask

                feats_source_domain = feats_source_inn - feats_source_clas
                feats_target_domain = feats_target_inn - feats_target_clas

                con_loss_iter += con_loss

                # reconstruction the feature via swap startegy class

                feats_source_swapc = feats_target_clas + feats_source_domain
                feats_target_swapc = feats_source_clas + feats_target_domain

                con_loss = self.cdd.forward([feats_source_class.squeeze()], [feats_target_class.squeeze()],
                                            source_nums_cls, target_nums_cls)[self.discrepancy_key] \
                           + self.cdd.forward([feats_source_swapc.squeeze()], [feats_target_swapc.squeeze()],
                                            source_nums_cls, target_nums_cls)[self.discrepancy_key]
                swap += con_loss

                feats_source_rec = self.INN(feats_source_swapc, rev=True)
                feats_target_rec = self.INN(feats_target_swapc, rev=True)
                probs_source_rec = nn.Softmax(dim=1)(self.C(feats_source_rec.squeeze()))
                probs_target_rec = nn.Softmax(dim=1)(self.C(feats_target_rec.squeeze()))

                swap_loss += self.cdd.forward([feats_source_rec.squeeze()], [feats_source.squeeze()],
                                            source_nums_cls, source_nums_cls)[self.discrepancy_key] \
                           + self.cdd.forward([feats_target_rec.squeeze()], [feats_target.squeeze()],
                                              target_nums_cls, target_nums_cls)[self.discrepancy_key] \
                           + self.cdd.forward([probs_source_rec.squeeze()], [probs_source.squeeze()],
                                              source_nums_cls, source_nums_cls)[self.discrepancy_key] \
                           + self.cdd.forward([probs_target_rec.squeeze()], [probs_target.squeeze()],
                                              target_nums_cls, target_nums_cls)[self.discrepancy_key]

                inn_loss_iter += con_loss

                # sample random variable
                feats_norm = to_cuda(torch.FloatTensor(feats_source_domain.size()).normal_()) * (1 - mask)

                feats_source_norm_re = feats_source_clas + feats_norm
                feats_target_norm_re = feats_target_clas + feats_norm

                feats_source_norm_inn = self.INN(feats_source_norm_re, rev=True)
                feats_target_norm_inn = self.INN(feats_target_norm_re, rev=True)
                probs_source_norm_inn = nn.Softmax(dim=1)(self.C(feats_source_norm_inn.squeeze()))
                probs_target_norm_inn = nn.Softmax(dim=1)(self.C(feats_target_norm_inn.squeeze()))

                feats_source_norm_inn = feats_source_norm_inn.detach()
                feats_target_norm_inn = feats_target_norm_inn.detach()
                probs_source_norm_inn = probs_source_norm_inn.detach()
                probs_target_norm_inn = probs_target_norm_inn.detach()

                align_loss = self.cdd.forward([feats_source.squeeze()], [feats_source_norm_inn.squeeze()],
                                                 source_nums_cls, source_nums_cls)[self.discrepancy_key] \
                                + self.cdd.forward([probs_source.squeeze()], [probs_source_norm_inn.squeeze()],
                                                   source_nums_cls, source_nums_cls)[self.discrepancy_key] \
                                + self.cdd.forward([feats_target.squeeze()], [feats_target_norm_inn.squeeze()],
                                                   target_nums_cls, source_nums_cls)[self.discrepancy_key] \
                                + self.cdd.forward([probs_target.squeeze()], [probs_target_norm_inn.squeeze()],
                                                   target_nums_cls, source_nums_cls)[self.discrepancy_key]


                total_loss = align_loss * self.opt.HYPER.ALIGN_WEIGHT + swap_loss * self.opt.HYPER.SWAP_WEIGHT
                
                total_loss.backward()
                loss += total_loss

            self.optimizer['I'].step()
            self.optimizer['F'].step()
            self.optimizer['C'].step()
            # model eval and ckpt save
            save_flag = False

            if self.opt.TRAIN.LOGGING and (update_iters + 1) % \
                    (max(1, self.iters_per_loop // self.opt.TRAIN.NUM_LOGGING_PER_LOOP)) == 0:
                accu = self.model_eval(source_logits, source_gt)
                cur_loss = {'ce_loss': ce_loss, 'con_loss': con_loss, 'align_loss': align_loss,
                            'swap_loss': swap_loss, 'total_loss': loss}
                self.logging(cur_loss, accu)

            self.opt.TRAIN.TEST_INTERVAL = min(1.0, self.opt.TRAIN.TEST_INTERVAL)
            self.opt.TRAIN.SAVE_CKPT_INTERVAL = min(1.0, self.opt.TRAIN.SAVE_CKPT_INTERVAL)

            if self.opt.TRAIN.TEST_INTERVAL > 0 and \
                    (update_iters + 1) % int(self.opt.TRAIN.TEST_INTERVAL * self.iters_per_loop) == 0:
                with torch.no_grad():
                    self.net.module.set_bn_domain(self.bn_domain_map[self.target_name])
                    accu = self.test()
                    if accu > self.max_accu:
                        self.max_accu = accu
                        save_flag = True
                    print('Test at (loop %d, iters: %d) with %s: %.4f. , max: %.4f. , flag:%s' % (self.loop,
                                                                                                  self.iters,
                                                                                                  self.opt.EVAL_METRIC,
                                                                                                  accu, self.max_accu,
                                                                                                  save_flag))
                    

            if self.opt.TRAIN.SAVE_CKPT_INTERVAL > 0 and save_flag and \
                    (update_iters + 1) % int(self.opt.TRAIN.SAVE_CKPT_INTERVAL * self.iters_per_loop) == 0:
                self.save_ckpt()

            update_iters += 1
            self.iters += 1

            # update stop condition
            if update_iters >= self.iters_per_loop:
                stop = True
            else:
                stop = False

    def to_prototype(self, w, x):
        y = to_cuda(torch.zeros((len(x), 2048)))
        for i in range(len(x)):
            y[i] = w[x[i]]
        return y.unsqueeze(2).unsqueeze(3)

    def ConLoss(self, x, y):
        conLoss = self.MSELoss(x, y)
        # conLoss = self.KLoss(x, y)
        # conLoss = 1 - torch.mean(self.sim(x.squeeze(), y.squeeze()))
        return conLoss

    def inloss(self, x, y, alpha=0.05):
        inloss = 0
        for i in range(len(x)):
            tmp, _ = hsic_gam(x[i], y[i], alpha)
            inloss += tmp

        return inloss

    def get_mask(self, x, gt):
        x_new = x.clone().detach()
        x_new = Variable(x_new.data, requires_grad=True)

        out = self.C(x_new.squeeze())
        class_num = out.shape[1]
        num_rois = x_new.shape[0]
        num_channel = x_new.shape[1]

        one_hot = torch.zeros((1), dtype=torch.float32).cuda()
        one_hot = Variable(one_hot, requires_grad=False)
        sp_i = torch.ones([2, num_rois]).long()
        sp_i[0, :] = torch.arange(num_rois)
        sp_i[1, :] = gt[0:num_rois]
        sp_v = torch.ones([num_rois])

        one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
        one_hot_sparse = Variable(one_hot_sparse, requires_grad=False)
        one_hot = torch.sum(out * one_hot_sparse)
        self.net.eval()
        self.net.zero_grad()
        one_hot.backward()

        grads_val = x_new.grad.clone().detach()
        channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2)

        self.net.zero_grad()
        self.net.train()
        # channel_mean = (channel_mean.view(num_rois, num_channel, 1, 1) * x).squeeze()
        vector_thresh_percent = 1024
        vector_thresh_value = torch.sort(channel_mean, dim=1, descending=True)[0][:, vector_thresh_percent]
        vector_thresh_value = vector_thresh_value.view(num_rois, 1).expand(num_rois, num_channel)
        vector = torch.where(channel_mean> vector_thresh_value,
                             torch.ones(channel_mean.shape).cuda(),
                             torch.zeros(channel_mean.shape).cuda())
        tmp_mask = vector.view(num_rois, num_channel, 1, 1)
        for i in range(len(gt)):
             self.mask[int(gt[i])] = tmp_mask[i]
        return tmp_mask


    def get_gt_mask(self, mask, gt):
        gt_mask = torch.ones(len(gt), 2048, 1, 1).cuda()
        for i in range(len(gt)):
            gt_mask[i] *= mask[int(gt[i])]
        return gt_mask

    def RCELoss(self, pred, labels):
        pred = F.softmax(pred, dim=1).cuda()
        pred = torch.clamp(pred, min=1e-4, max=1.0)
        label_one_hot = F.one_hot(labels, self.opt.DATASET.NUM_CLASSES).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        loss = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        return 0.3 * loss.mean()

    def LDA_dimensionality(self, x, y):

        label_ = list(set(y))

        x_classify = {}

        for label in label_:
            x1 = np.array(x[i] for i in range(len(x)) if y[i] == label)
            x_classify[label] = x1

        mju = np.mean(x, axis=0)
        mju_classify = {}

        for label in label_:
            mju1 = np.mean(x_classify[label], axis=0)
            mju_classify[label] = mju1

        # St = np.dot((X - mju).T, X - mju)

        Sw = np.zeros((len(mju), len(mju)))  # 计算类内散度矩阵
        for i in label_:
            Sw += np.dot((x_classify[i] - mju_classify[i]).T,
                         x_classify[i] - mju_classify[i])

        # Sb=St-Sw

        Sb = np.zeros((len(mju), len(mju)))  # 计算类内散度矩阵
        for i in label_:
            Sb += len(x_classify[i]) * np.dot((mju_classify[i] - mju).reshape(
                (len(mju), 1)), (mju_classify[i] - mju).reshape((1, len(mju))))

        eig_vals, eig_vecs = np.linalg.eig(
            np.linalg.inv(Sw).dot(Sb))  # 计算Sw-1*Sb的特征值和特征矩阵

        sorted_indices = np.argsort(eig_vals)
        topk_eig_vecs = eig_vecs[:, sorted_indices[:-k - 1:-1]]  # 提取前k个特征向量
        return topk_eig_vecs
   