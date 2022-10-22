
from mylib.data.data_loader import DataLoader_noise
import argparse
import torchvision.transforms as transforms
import mylib.models as models
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import types
import os
import torch
import numpy as np
import tools

def create_dir(args):
    save_dir = args.save_dir + '/' + args.dataset + '/' + '%s' % (args.noise_type) + '/' + 'noise_rate_%s' % (
        args.noise_rate) + '/' + 'lam=%f6_' % (args.lam) + '%d' %args.seed
    if not os.path.exists(save_dir):
        os.system('mkdir -p %s' % (save_dir))

    model_dir = save_dir + '/models'

    if not os.path.exists(model_dir):
        os.system('mkdir -p %s' % (model_dir))

    matrix_dir = save_dir + '/matrix'

    if not os.path.exists(matrix_dir):
        os.system('mkdir -p %s' % (matrix_dir))

    logs = open(save_dir + '/log.txt', 'w')

    return  save_dir, model_dir, matrix_dir, logs


def run_dnl(train_data, val_data, test_data, noise_type, noise_rate, dataset, n_epoch,
    num_classes, device,arch):

    

    volargs = types.SimpleNamespace()

    volargs.lr=0.01
    volargs.save_dir='saves_m'
    volargs.dataset= dataset
    volargs.n_epoch=n_epoch
    volargs.num_classes = num_classes
    volargs.noise_type = noise_type
    volargs.noise_rate = noise_rate
    volargs.seed=1
    volargs.batch_size=128
    volargs.device=device
    volargs.weight_decay=1e-4
    volargs.lam=0.000
    volargs.arch = arch
    milestones = [30,60]

  
    model = models.__dict__[volargs.arch](feature_dim=len(train_data.dataset.data[0]), num_classes=volargs.num_classes)
    trans = models.__dict__["sig_t"](volargs.device, volargs.num_classes, init=2)
    optimizer_trans = optim.SGD(model.parameters(), lr=volargs.lr, weight_decay=volargs.weight_decay, momentum=0.9)
    save_dir, model_dir, matrix_dir, logs = create_dir(volargs)

    print(volargs)


    #data_loader
    train_loader = DataLoader_noise(dataset=train_data, 
                            batch_size=volargs.batch_size,
                            shuffle=True,
                            num_workers=4,
                            drop_last=False)

    val_loader = DataLoader_noise(dataset=val_data,
                            batch_size=volargs.batch_size,
                            shuffle=False,
                            num_workers=4,
                            drop_last=False)

    test_loader = DataLoader_noise(dataset=test_data,
                            batch_size=volargs.batch_size,
                            num_workers=4,
                            drop_last=False)

    #optimizer and StepLR
    optimizer_es = optim.Adam(trans.parameters(), lr=volargs.lr, weight_decay=0)
    scheduler1 = MultiStepLR(optimizer_es, milestones=milestones, gamma=0.1)
    scheduler2 = MultiStepLR(optimizer_trans, milestones=milestones, gamma=0.1)


    loss_func_ce = F.nll_loss

    #cuda
    if torch.cuda.is_available:
        model = model.cuda()
        trans = trans.cuda()

    val_loss_list = []
    val_acc_list = []
    test_acc_list = []

    t = trans()
    est_T = t.detach().cpu().numpy()


    for epoch in range(volargs.n_epoch):

        print('epoch {}'.format(epoch + 1), file=logs,flush=True)
        model.train()
        trans.train()

        train_loss = 0.
        train_vol_loss =0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.
        eval_loss = 0.
        eval_acc = 0.

        for  i, (batch_x, batch_y, indexes,  _, _) in enumerate(train_loader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

            optimizer_es.zero_grad()
            optimizer_trans.zero_grad()



            clean = model(batch_x)
            if epoch > 20:
                clean = F.gumbel_softmax(clean, 1)
            else:
                clean = F.gumbel_softmax(clean, 1)
            t = trans()

            out = torch.mm(clean, t)     
            vol_loss = t.slogdet().logabsdet
            ce_loss = loss_func_ce(out.log(), batch_y.long())
            loss = ce_loss + volargs.lam * vol_loss      
            train_loss += loss.item()
            train_vol_loss += vol_loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            loss.backward()
            optimizer_es.step()
            optimizer_trans.step()

        scheduler1.step()
        scheduler2.step()

        with torch.no_grad():
            model.eval()
            trans.eval()
            for batch_x, batch_y, clean_target,  _, _  in val_loader:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                clean = model(batch_x)
                if epoch > 20:
                    clean = F.gumbel_softmax(clean, 1)
                else:
                    clean = F.gumbel_softmax(clean, 1)
                t = trans()

                out = torch.mm(clean, t)
                loss = loss_func_ce(out.log(), batch_y.long())
                val_loss += loss.item()
                pred = torch.max(out, 1)[1]
                val_correct = (pred == batch_y).sum()
                val_acc += val_correct.item()


        with torch.no_grad():
            model.eval()
            trans.eval()

            for batch_x, batch_ys , clean_target, batch_y, _ in test_loader:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                clean = model(batch_x)
                if epoch > 20:
                    clean = F.gumbel_softmax(clean, 1)
                else:
                    clean = F.gumbel_softmax(clean, 1)
                loss = loss_func_ce(clean.log(), batch_y.long())
                eval_loss += loss.item()
                pred = torch.max(clean, 1)[1]
                eval_correct = (pred == batch_y).sum()
                eval_acc += eval_correct.item()

            est_T = t.detach().cpu().numpy()
      
            matrix_path = matrix_dir + '/' + 'matrix_epoch_%d.npy' % (epoch+1)
            np.save(matrix_path, est_T)


        val_loss_list.append(val_loss / (len(val_data)))
        val_acc_list.append(val_acc / (len(val_data)))
        test_acc_list.append(eval_acc / (len(test_data)))


    val_loss_array = np.array(val_loss_list)
    val_acc_array = np.array(val_acc_list)
    model_index = np.argmin(val_loss_array)
    model_index_acc = np.argmax(val_acc_array)

    matrix_path = matrix_dir + '/' + 'matrix_epoch_%d.npy' % (model_index)
    final_est_T = np.load(matrix_path)
    print(final_est_T)

    print("Final test accuracy: %f" % test_acc_list[model_index], file=logs,flush=True)
    print("Best epoch: %d" % model_index)
    logs.close()
    return final_est_T, test_acc_list[model_index_acc]

