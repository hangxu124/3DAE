import torch
from torch.autograd import Variable
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve


from utils import *


def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))
    val_loader  = data_loader
    #val_loader_n1 = data_loader[1]
    buffer_p = []
    buffer_t = []
    prob = np.zeros((1,2))
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    aloss1 = AverageMeter()
    aloss2 = AverageMeter()

    end_time = time.time()
    for i, ((inputs1, targets1)) in enumerate(val_loader):
        data_time.update(time.time() - end_time)
        #print (":::::::::::::::",inputs1.size())

        if not opt.no_cuda:
            targets1 = targets1.cuda()
            #targets2 = targets2.cuda()
            inputs1  = inputs1.cuda()
            #inputs2  = inputs2.cuda()
        with torch.no_grad():
            inputs1  = Variable(inputs1)
            targets1 = Variable(targets1)
            #inputs2  = Variable(inputs2)
            #targets2 = Variable(targets2)

        rec = model(inputs1, score=False)
        outputs = model(inputs1, score=True)
        #print (outputs)
        targets = targets1

        loss1 = criterion[0](outputs, targets) 
        loss2 = criterion[1](rec, inputs1) 

        alpha = 1
        beta  = 0 #0.04

        loss = loss1 * alpha + loss2 * beta 
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,2))
        prob1, p,t = save_output(outputs.data, targets.data, topk=(1,))
        prob=np.concatenate((prob,prob1),axis=0)
        #print (p,t)
        buffer_p.extend(p)
        buffer_t.extend(t)

        #prec5 =100
        losses.update(loss.data, inputs1.size(0))
        aloss1.update(loss1.data, inputs1.size(0))
        aloss2.update(loss2.data, inputs1.size(0))
        top1.update(prec1, inputs1.size(0))
        top5.update(prec5, inputs1.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i % 100 ==0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
                  'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
                  'Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                      epoch,
                      i + 1,
                      len(val_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      loss1=aloss1,
                      loss2=aloss2,
                      top1=top1,
                      top5=top5))

    logger.log({'epoch': epoch,
                'loss': losses.avg.item(),
                'loss1': aloss1.val.item(),
                'loss2': aloss2.val.item(),
                'prec1': top1.avg.item(),
                'prec5': top5.avg.item()})
    new_p = np.array(buffer_p).reshape(-1,1)
    new_t = np.array(buffer_t).reshape(-1,1)
    np.save("buffer_p.npy",new_p)
    np.save("buffer_t.npy",new_t)
    np.save("prob_mv2_front_depth.npy",prob)
    cm = confusion_matrix(new_t,new_p)
    print (cm)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    print (cm_normalized)
    #print (len(buffer_p))
    #rint (buffer_p)
    # with open("buffer_p.txt","a") as bp:
    #   bp.write(str(buffer_p))
    # with open("buffer_t.txt","a") as bt:
    #   bt.write(str(buffer_t))
    # #np.savetxt("buffer_t.txt",buffer_t)
    return losses.avg.item(), top1.avg.item()