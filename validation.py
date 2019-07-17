import torch
from torch.autograd import Variable
import time
import sys

from utils import *
import numpy as np

def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))
    val_loader  = data_loader
    #val_loader_n = data_loader[1]

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    aloss1 = AverageMeter()
    aloss2 = AverageMeter()

    end_time = time.time()
    for i, (inputs1, targets1) in enumerate(val_loader):
        data_time.update(time.time() - end_time)

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
        #inputs1_t = inputs1[:,:2,:,:,:]
        # s1 = inputs1[0,0,0,:,:].cpu().numpy()
        # s2 = inputs1[0,1,0,:,:].cpu().numpy()
        # np.save("s1.npy",s1)
        # np.save("s2.npy",s2)

        #inputs1_f = inputs1[:,2:,:,:,:]
        rec_t,feature1_t = model (inputs1,f=0,score=False)
        #rec_f,feature1_f = model (inputs1_f,f=0,score=False)
        # feature1 = torch.cat((feature1_t,feature1_f),dim=1)
        #feature2,_ = model(inputs2, score=False)
        #outputs = model(inputs1, score=True)
        #feature1 = feature1_t + feature1_f
        outputs = model(x=feature1_t, f=feature1_t, score=True, fusion=1)
        targets = targets1
        #print ("#########################################")
        #print(feature1_t.size(), rec_f.size())
        #print (feature1.size(),outputs.size(),targets.size())
        #print ("##########################################")

        loss1 = criterion[0](outputs, targets) 
        loss2 = criterion[1](rec_t, inputs1) # + criterion[1](rec_f, inputs1_f) 

        alpha = 1
        beta  = 0.01

        loss = loss1 * alpha + loss2 * beta 
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,2))

        losses.update(loss.data, inputs1.size(0))
        aloss1.update(loss1.data, inputs1.size(0))
        aloss2.update(loss2.data, inputs1.size(0))
        top1.update(prec1, inputs1.size(0))
        top5.update(prec5, inputs1.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i % 10 ==0:
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

    return losses.avg.item(), top1.avg.item()