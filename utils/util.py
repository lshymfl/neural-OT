import torch
from  torch import nn
from torchvision.utils import save_image
from torch.autograd import Variable
import fnmatch
import os
import scipy.io as sio
from utils.Regularization import Regularization 
from score.both import get_inception_and_fid_score
import numpy as np
import copy
 
import torch.nn.functional as F
from utils.module import *
import lpips
#from scipy.stats import shapiro   ######added 
from utils.brenier import BrenierHeightNet, init_optimizer
from tqdm import trange, tqdm 
import time

def train_ae(model, trainloader,testloader,args,resume=False):
    ''' train AE model '''
    recnums = args.recnums 
    rows = int(np.sqrt(recnums))

    for test_data in testloader:
        if args.mdb:
            test_img, _ = test_data
        else:
            test_img, _, _ = test_data
        break
    ###############################add
    for data_data in trainloader:
        if args.mdb:
            data_img, _ = data_data
        else:
            data_img, _, _ = data_data
        break
       
    ae_model_path = os.path.join(args.result_root_path,'ae_models')
    if resume:
        for file in os.listdir(ae_model_path):
            print(file)
            model.load_state_dict(torch.load(os.path.join(ae_model_path, file)))
    ### add regularization
    if args.wd>0:
        if args.p==1:
            reg_loss=Regularization(model, args.wd, p=1)
            print("L1 regularization")
        else:
            reg_loss=Regularization(model, args.wd, p=2)
            print("L2 regularization") 
    else:
        print("no regularization")
    ########################
    criterion = WeightedLoss()
    mse_fn = nn.MSELoss()
    loss_fn = lpips.LPIPS(net='alex').cuda() # best forward scores
 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr , weight_decay=args.weight_decay)
    #### show model size
    model_size = 0
    for param in model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024)) 

    #save input test image
    img_save_path = os.path.join(args.result_root_path,'rec_imgs')
    save_image(test_img[:recnums], os.path.join(img_save_path, 'test_image_input.jpg'),nrow=rows)
    save_image(data_img[:recnums], os.path.join(img_save_path, 'data_image_input.jpg'),nrow=rows)  ###########################add
    for epoch in range(args.epochs):
        count_train = 0
        loss_train = 0.0
        count_test = 0
        loss_test = 0.0
        for data in trainloader:
            
            if args.mdb:
                img, _ = data
            else:
                img, _,_ = data

            img = Variable(img).cuda()         
            # ===================forward=====================
            output,_ = model(img)
            mse_loss = mse_fn(output, img)
            lpips_loss = loss_fn(output, img).mean()

            if args.wd > 0 and args.lpips_weight == 0:
                loss = mse_loss + reg_loss(model)/args.batch_size #feat_loss + 
            elif args.wd == 0 and args.lpips_weight > 0:
                loss = (1-args.lpips_weight)*mse_loss + args.lpips_weight*lpips_loss #criterion(mse_loss, lpips_loss )  
            else:
                loss = mse_loss  

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            count_train += 1
 
            # ===================log========================
            #print('epoch [{}/{}], loss1:{:.4f}, loss2:{:.4f}'.format(epoch, args.epochs, loss1.item(), loss2.item()))
                  
        if args.testload:
            for data in testloader:
                if args.mdb:
                    img, _ = data
                else:
                    img, _,_ = data  
                img = Variable(img).cuda()
                #model.eval()
                with torch.no_grad():
                    output,_ = model(img)
                mse_loss = mse_fn(output, img)
                lpips_loss = loss_fn(output, img).mean()
                if args.wd > 0 and args.lpips_weight == 0:
                    loss = mse_loss + reg_loss(model)/args.batch_size 
                elif args.wd == 0 and args.lpips_weight > 0:
                    loss = (1-args.lpips_weight)*mse_loss + args.lpips_weight*lpips_loss   
                else:
                    loss = mse_loss
    
                loss_test += loss.item()
                count_test += 1
            loss_test /= count_test
        else:
            loss_test = 0

        loss_train /= count_train
 
        print('epoch [{}/{}], loss_train:{:.8f}, loss_test:{:.8f}'.format(epoch, args.epochs, loss_train, loss_test))#add
        
        if args.testload:
            testput,_ = model(test_img.cuda())
            pic = testput.data.cpu()
            save_image(pic[:recnums], os.path.join(img_save_path, 'Epoch_{}_test_image_{:06f}_{:06f}.jpg'.format(epoch, loss_train, loss_test)),nrow=rows)
        ###################################add
        dataput,_ = model(data_img.cuda())
        datapic = dataput.data.cpu()
        save_image(datapic[:recnums], os.path.join(img_save_path, 'Epoch_{}_data_image_{:06f}_{:06f}.jpg'.format(epoch, loss_train, loss_test)),nrow=rows)
        ###################################
        torch.save(model.state_dict(), os.path.join(ae_model_path,'Epoch_{}_sim_autoencoder_{:06f}_{:06f}.pth'.format(epoch, loss_train, loss_test)))

def fine_tuning_ae(model, trainloader,testloader,ae_model_path,args,resume=False):       
    ''' fine-tuning refine train AE model '''
    params = args.encoder_param
    for param in model.encoder.parameters():
        param.requires_grad = params
    for param in model.decoder.parameters():
        param.requires_grad = True

    recnums = args.recnums 
    rows = int(np.sqrt(recnums))
    for data_data in trainloader:
        if args.mdb:
            data_img, _ = data_data
        else:
            data_img, _, _ = data_data   
        break
    for test_data in testloader:
        if args.mdb:
            test_img, _ = test_data
        else:
            test_img, _, _ = test_data
        break
    ae_model_path = ae_model_path
    if resume:
        for file in os.listdir(ae_model_path):
            print(file)
            model.load_state_dict(torch.load(os.path.join(ae_model_path, file)))

    criterion = WeightedLoss()
    mse_fn = nn.MSELoss()
    loss_fn = lpips.LPIPS(net='alex').cuda() # best forward scores
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr , weight_decay=args.weight_decay) 
    #save input test image
    img_save_path = os.path.join(args.result_root_path,'rec_imgs')
    save_image(data_img[:recnums], os.path.join(img_save_path, 'data_image_input.jpg'),nrow=rows)
   
    for epoch in range(args.epochs):
        count_train = 0
        loss_train = 0.0
        loss_test = 0.0
        for data in trainloader:
            if args.mdb:
                img, _ = data
            else:
                img, _,_ = data

            img = Variable(img).cuda()
            # ===================forward=====================
            output,_ = model(img)
            mse_loss = mse_fn(output, img)
            lpips_loss = loss_fn(output, img).mean()
            if args.wd > 0 and args.lpips_weight == 0:
                loss = mse_loss + reg_loss(model)/args.batch_size #feat_loss + 
            elif args.wd == 0 and args.lpips_weight > 0:
                loss = (1-args.lpips_weight)*mse_loss + args.lpips_weight*lpips_loss  
            else:
                loss = mse_loss
            
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ===================log========================
            #print('epoch [{}/{}], loss:{:.4f}'.format(epoch, args.epochs, loss.item()))
            loss_train += loss.item()
            count_train += 1

        loss_train /= count_train
        loss_test = 0

        print('epoch [{}/{}], loss_train:{:.8f}, loss_test:{:.8f}'.format(epoch, args.epochs, loss_train, loss_test))#add
 
        output,_ = model(data_img.cuda())
        pic = output.data.cpu()
        save_image(pic[:recnums], os.path.join(img_save_path, 'Epoch_{}_train_image_{:.6f}.jpg'.format(epoch,loss_train)),nrow=rows)
        torch.save(model.state_dict(), os.path.join(ae_model_path,'Epoch_{}_sim_refine_autoencoder_{:06f}_{:06f}.pth'.format(epoch, loss_train, loss_test)))


def extract_feature_ae(model, dataloader,ae_model_path,args,data_len):  
    ''' extract_feature of AE model '''
    features = torch.empty([data_len, args.dim_z], dtype=torch.float, requires_grad=False, device='cpu')
    ae_model_path = ae_model_path
    for file in os.listdir(ae_model_path):
        print(file)
        model.load_state_dict(torch.load(os.path.join(ae_model_path, file)))
    i = 0
    for data in dataloader:
        if args.mdb:
            img, _ = data
        else:
            img, _,_ = data
        img = img.cuda()
        img.requires_grad = False
        # ===================forward=====================
        z = model.encoder(img.detach())
        features[i:i+img.shape[0], :] = z.squeeze().detach().cpu()   #.squeeze()
        i += img.shape[0]
    #print('Extracted {}/{} features...'.format(i, data_len))
    print('Extracted features complete')
    features = features[:i]
    feature_save_path = os.path.join(args.result_root_path,'ae_features.pt')
    torch.save(features, feature_save_path)
    print(features.shape)

def train_ot_model(h_model_path, model_net, alter_target_measure, idxs, args):
    if args.train:
        target_sample_path = os.path.join(args.result_root_path,'Y_sample.pt')
        target_measure_path = os.path.join(args.result_root_path,'nu_sample.pt')
        height_vector_path = os.path.join(args.result_root_path,'h_sample.pt')

        target_sample = torch.load(target_sample_path).cuda()   ###  cpu
        target_measure = torch.load(target_measure_path).cuda()
        height_vector = torch.load(height_vector_path).cuda()

        num_samples = target_sample.size(0)
        print(num_samples)
        num_train_samples = int(args.train_ratio * num_samples)
        num_test_samples = num_samples - num_train_samples

        #torch.manual_seed(123)
        indices = torch.randperm(num_samples)
        train_indices = indices[:num_train_samples]
        test_indices = indices[num_train_samples:]

        target_sample_train, target_measure_train, height_vector_train = target_sample[train_indices], target_measure[train_indices], height_vector[train_indices]
        target_sample_test, target_measure_test, height_vector_test = target_sample[test_indices], target_measure[test_indices], height_vector[test_indices]

        model_net.train()
        mse_fn = nn.MSELoss()   
        optimizer = torch.optim.Adam(model_net.parameters(), lr=args.lr_net, weight_decay=args.weight)#SGD/Adam/RMSprop
        #optimizer, scheduler = init_optimizer(model_net, args.lr, args.netepochs)

        # save loss data of train models
        train_losses = []
        test_losses = []
        start = time.time()
        for epoch in range(args.netepochs):

            count_train = 0
            loss_train = 0.0
            index = torch.randperm(target_sample_train.size(0))
            target_sample_train = target_sample_train[index]
            target_measure_train = target_measure_train[index]
            height_vector_train = height_vector_train[index]
            for i in trange(0, int(target_sample_train.size(0)/args.netbatch_size), 1, desc='train model'):
                batch_sample = target_sample_train[i*args.netbatch_size:(i+1)*args.netbatch_size]
                batch_measure = target_measure_train[i*args.netbatch_size:(i+1)*args.netbatch_size]
                batch_h = height_vector_train[i*args.netbatch_size:(i+1)*args.netbatch_size]

                optimizer.zero_grad()
                #print(batch_sample.shape)
                #print(batch_measure.shape)
                out_h = model_net(batch_sample, batch_measure.unsqueeze(1)).squeeze()
                loss = mse_fn(out_h, batch_h)
                #loss = loss + args.weight * torch.norm(out_h, 2)
               
                loss.backward()
                optimizer.step()
                #scheduler.step()

                loss_train += loss.item()
                count_train += 1
            
            loss_train /= count_train
            train_losses.append(loss_train)

            ## test models
            with torch.no_grad():
                #print(target_sample_test.shape)
                #print(target_measure_test.shape)
                out_h_test = model_net(target_sample_test, target_measure_test.unsqueeze(1)).squeeze()
                #out_h_test = model_net(target_sample_test, args.netbatch_size) 
            test_loss = mse_fn(out_h_test, height_vector_test)
            test_loss = test_loss.cpu().numpy()
            test_losses.append(test_loss) 
            #test_loss = 0
             

            print('epoch [{}/{}], loss_train:{:.6f}, loss_test:{:.6f}'.format(epoch, args.netepochs, loss_train, test_loss) )
            torch.save(model_net.state_dict(), os.path.join(h_model_path,'Epoch_{}_models.pth'.format(epoch, loss_train)) )
        
        end = time.time()
        print("Train_ot done at %.3f seconds." % (end - start)) 
        train_loss_data = np.array(train_losses)
        test_loss_data = np.array(test_losses)
        #print('test_loss_data=',test_loss_data, type(test_loss_data))
        data_dict = {'arr1': train_loss_data, 'arr2': test_loss_data}
        mat_path = os.path.join(args.result_root_path,'train_test_loss.mat')
        sio.savemat(mat_path, data_dict)

    else:
        if args.test_trainsample:
            target_sample_path = os.path.join(args.result_root_path,'Y_sample.pt')
            target_measure_path = os.path.join(args.result_root_path,'nu_sample.pt')
            height_vector_path = os.path.join(args.result_root_path,'h_sample.pt')
            target_sample = torch.load(target_sample_path).cuda() 
            target_measure = torch.load(target_measure_path).cuda()
            height_vector = torch.load(height_vector_path).cuda()
        else:
            target_sample_path = os.path.join(args.result_root_path,'ae_features.pt')
            tag_samples = torch.load(target_sample_path)
            if args.one_attribute:
                gr = tag_samples[idxs[0]]
                lr = tag_samples[idxs[1]]
                alter_tag_samples = torch.cat((gr,lr), dim=0)
            elif args.two_attribute:
                maglss = tag_samples[idxs[0]]
                manonglass = tag_samples[idxs[1]]
                femaglass = tag_samples[idxs[2]]
                femanonglass = tag_samples[idxs[3]]
                alter_tag_samples = torch.cat((maglss,manonglass,femaglass,femanonglass), dim=0)
            else:
                MBY = tag_samples[idxs[0]]
                MBNY = tag_samples[idxs[1]]
                MNBY = tag_samples[idxs[2]]
                MNBNY = tag_samples[idxs[3]]
                FMBY = tag_samples[idxs[4]]
                FMBNY = tag_samples[idxs[5]]
                FMNBY = tag_samples[idxs[6]]
                FMNBNY = tag_samples[idxs[7]]
                alter_tag_samples = torch.cat((MBY,MBNY,MNBY,MNBNY,FMBY,FMBNY,FMNBY,FMNBNY), dim=0)
            target_sample = alter_tag_samples.cuda()


            if args.alter_measture:
                target_measure = alter_target_measure.cuda()
            else:
                target_measure = torch.div(torch.ones(target_sample.size(0)), target_sample.size(0)).cuda()
            print(target_measure)
            
            height_vector_path = os.path.join(args.result_root_path,'hori.pt')
            height_vector = torch.load(height_vector_path).cuda()
            

        model_net.load_state_dict(torch.load(args.hmodel))
        h = torch.empty([target_sample.size(0)])
        lens = int(target_sample.size(0)/args.nums_batch)
        for i in range(args.nums_batch):
            target_sample0 = target_sample[i*lens:(i+1)*lens]
            target_measure0 = target_measure[i*lens:(i+1)*lens]
            h_batch = model_net(target_sample0, target_measure0.unsqueeze(1))
            #print(h_batch)
            h[i*lens:(i+1)*lens] = Variable(h_batch.squeeze().cpu() ,requires_grad=False)
        #h = Variable(h,requires_grad=False)
        torch.save(h, os.path.join(args.result_root_path, 'h_pred.pt') )

        ori_h = height_vector.detach().cpu().numpy()
        print('h_original=',ori_h.shape, ori_h)
        predh = h.detach().cpu().numpy()
        print('h_predicate=',predh.shape, predh)

        data_dict = {'arr1': ori_h, 'arr2': predh}
        mat_path = os.path.join(args.result_root_path,'trueH_predH.mat')
        sio.savemat(mat_path, data_dict)


def decode_feature(ae_model_path,model,args): 
    ''' decode new latent feature '''    
    for file in os.listdir(ae_model_path):
        print(file)  #if fnmatch.fnmatch(file, 'Epoch_*_sim__refine_autoencoder*.pth'): #or use Epoch_*_sim_refine_autoencoder*.pth
        model.load_state_dict(torch.load(os.path.join(ae_model_path, file)))
    gen_feature_path = os.path.join(args.result_root_path,'gen_features.mat')
    feature_dict = sio.loadmat(gen_feature_path)
    features = feature_dict['features']
    ids = feature_dict['ids']
           
    #=====================generate generated image pairs===========
    gen_im_pair_path = os.path.join(args.result_root_path,'gen_img_pairs')
    gen_im_path = os.path.join(args.result_root_path,'gen_imgs')
    
    num_feature = features.shape[0]
    z = torch.from_numpy(features).cuda()
    print(z.shape)
    z = z.view(num_feature,-1,1,1)

       
    k = int(args.max_gen_samples/args.num_m)
    print(args.max_gen_samples)
    for i in range(k):
        z1 = z[i*args.num_m:(i+1)*args.num_m,:,:,:]
        #z1 = z[i*args.num_m:(i+1)*args.num_m,:]
        with torch.no_grad():
            y1 = model.decoder(z1)
        print(y1.shape)
        for k in range(args.num_m):
            y_gen = y1[k,:,:,:]
            if args.max_gen_samples <= 1000:
                save_image(y_gen.cpu(), os.path.join(gen_im_pair_path, 'img_{0:07d}_gen.jpg'.format(k+i*args.num_m)))
            else:
                save_image(y_gen.cpu(), os.path.join(gen_im_path, 'img_{0:07d}_gen.jpg'.format(k+i*args.num_m)))        

    save_image(y1[0:36].cpu(), os.path.join(args.result_root_path, 'gen_imgs.jpg'), nrow=6)      

    '''y_all = y1
    nums = args.gennums
    rows = int(np.sqrt(nums))
    for i in range(3):
        save_image(y_all[i*nums:(i+1)*nums,:,:,:].cpu(), os.path.join(args.result_root_path, 'gen_img_{}.jpg'.format(i)), nrow=rows)'''

    '''
    with torch.no_grad():
        y1 = model.decoder(z)
    for k in range(y1.size(0)):
        y_gen = y1[k,:,:,:]
        save_image(y_gen.cpu(), os.path.join(gen_im_pair_path, 'img_{0:07d}_gen.jpg'.format(k)))
    save_image(y1.cpu(), os.path.join(args.result_root_path, 'gen_img.jpg'), nrow=10)'''

    