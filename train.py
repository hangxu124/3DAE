import torch
from torch.autograd import Variable
import time
import os
import sys
import numpy as np
from utils import *


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))
    train_loader_a  = data_loader[0]
    train_loader_n1 = data_loader[1]

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    aloss1 = AverageMeter()
    aloss2 = AverageMeter()

    end_time = time.time()
    for i, ((inputs1, targets1),(inputs2,targets2)) in enumerate(zip(train_loader_a,train_loader_n1)):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets1 = targets1.cuda()
            targets2 = targets2.cuda()
            inputs1  = inputs1.cuda()
            inputs2  = inputs2.cuda()
        inputs1 = Variable(inputs1)
        inputs2 = Variable(inputs2)
        # print (inputs1.size())
        # inputs1_t = inputs1[:,:2,:,:,:]
        # inputs1_f = inputs1[:,2:,:,:,:]
        # inputs2_t = inputs2[:,:2,:,:,:]
        # inputs2_f = inputs2[:,2:,:,:,:]

        #save1 = inputs1[0,0,0,:,:].cpu().numpy()
        #save2 = inputs1[0,1,0,:,:].cpu().numpy()
        # save3 = inputs1[0,2,0,:,:].cpu().numpy()
        # save4 = inputs1[0,3,0,:,:].cpu().numpy()
        # save5 = inputs2[0,0,0,:,:].cpu().numpy()
        # save6 = inputs2[0,1,0,:,:].cpu().numpy()
        # save7 = inputs2[0,2,0,:,:].cpu().numpy()
        # save8 = inputs2[0,3,0,:,:].cpu().numpy()
        #np.save("save1.npy",save1)
        #np.save("save2.npy",save2)
        # np.save("save3.npy",save3)
        # np.save("save4.npy",save4)
        # np.save("save5.npy",save5)
        # np.save("save6.npy",save6)
        # np.save("save7.npy",save7)
        # np.save("save8.npy",save8)

        targets1 = Variable(targets1)
        targets2 = Variable(targets2)

        rec_t, feature1t = model(inputs1, f=0, score=False)
        #rec_f, feature1f = model(inputs1_f, f=0, score=False)

        _, feature2t = model(inputs2, f=0, score=False)
        #_, feature2f = model(inputs2_f, f=0, score=False)

        #fea torch.cat((feature1,feature2), dim=1)
        #print ("#########################################")
        #print(feature1t.size(), rec_t.size())
        
        #feature1 = feature1t+feature1f
        #eature2 = feature2t+feature2f
        #feature1 = torch.cat((feature1t,feature1f), dim=1)
        #feature2 = torch.cat((feature2t,feature2f), dim=1)
        feature = torch.cat((feature1t,feature2t),dim=0)
        #print ("feature :",feature.size())

        #print (feature.size(),feature1.size(),feature2.size())
        #print ("##########################################")
        #print (f.size())
        #outputs = model(torch.cat((inputs1, inputs2), 0), score=True)
        outputs = model (x=feature, f=feature, score=True, fusion = 1)
        #print (outputs.size())
        targets = torch.cat((targets1, targets2), 0)
        #targets = targets1
        # print (outputs1)
        # print (outputs2)

        loss1 = criterion[0](outputs, targets) 
        loss2 = criterion[1](rec_t, inputs1) #+ criterion[1](rec_f,inputs1_f)

        alpha = 1
        beta  = 0.01

        loss = loss1 * alpha + loss2 * beta 
        
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,2))

        losses.update(loss.data, inputs1.size(0))
        aloss1.update(loss1.data, inputs1.size(0))
        aloss2.update(loss2.data, inputs1.size(0))
        top1.update(prec1, inputs1.size(0))
        top5.update(prec5, inputs1.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(train_loader_a) + (i + 1),
            'loss': losses.val.item(),
            'loss1': aloss1.val.item(),
            'loss2': aloss2.val.item(),
            'prec1': top1.val.item(),
            'prec5': top5.val.item(),
            'lr': optimizer.param_groups[0]['lr']
        })

        if i % 10 ==0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
                  'Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                      epoch,
                      i,
                      len(train_loader_a),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      loss1=aloss1,
                      loss2=aloss2,
                      top1=top1,
                      top5=top5,
                      lr=optimizer.param_groups[0]['lr']))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'loss1': aloss1.val.item(),
        'loss2': aloss2.val.item(),
        'prec1': top1.avg.item(),
        'prec5': top5.avg.item(),
        'lr': optimizer.param_groups[0]['lr']
    })

    #if epoch % opt.checkpoint == 0:
    #    save_file_path = os.path.join(opt.result_path,
    #                                  'save_{}.pth'.format(epoch))
    #    states = {
    #        'epoch': epoch + 1,
    #        'arch': opt.arch,
    #        'state_dict': model.state_dict(),
    #        'optimizer': optimizer.state_dict(),
    #    }
    #    torch.save(states, save_file_path)
