# -*- coding: utf-8 -*-

import os
import fnmatch
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import time
from modelsAE.HQ256 import autoencoder
from utils.P_loader import P_loader
from utils.util import *
from utils.ot_util import compute_ot
import numpy as np
from PIL import Image
import pandas as pd
 
#################fid #################
from score.fid_score import get_activations,  calculate_frechet_distance  ,compute_statistics_of_path
from score.inception import InceptionV3
from score.IS import inception_score
#################dataset #################
from datasets import get_dataset
from modelsOT.brenier2560 import  DualInputNet, DualInputNet1 

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.set_printoptions(precision=8)
def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ['yes', 'true', 't', 'y']:
        return True
    elif val.lower() in ['no', 'false', 'f', 'n']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ae", help="whether to train AE", dest='actions', action='append_const', const='train_ae')
    parser.add_argument("--refine_ae", help="whether to refine AE", dest='actions', action='append_const', const='refine_ae')
    parser.add_argument("--extract_feature", help="whether to extract latent code with AE encoder", dest='actions', action='append_const', const='extract_feature')
    parser.add_argument("--train_ot", help="whether to train (i.e. compute) OT with OT solver", dest='actions', action='append_const', const='train_ot')
    parser.add_argument("--train_net_ot", help="train ot by network", dest='actions', action='append_const', const='train_net_ot')
    parser.add_argument("--generate_feature", help="whether to generate new latent codes", dest='actions', action='append_const', const='generate_feature')
    parser.add_argument("--decode_feature", help="whether to decode generated latent codes", dest='actions', action='append_const', const='decode_feature')
    parser.add_argument("--calculate_fid", help="whether to decode generated latent codes", dest='actions', action='append_const', const='calculate_fid')
    parser.add_argument("--data_root_train", help='path to training set directory (for AE)', type=str, metavar="", dest="data_root_train", 
    default='/home/lishenghao/lshdata/2023data/celeba_HQ256/part')  #27000  /home/ubuntu/workspace/data/CelebA256/
    parser.add_argument("--data_root_test", help='path to testing set directory(for AE)', type=str, metavar="", dest= "data_root_test",
    default='/home/lishenghao/lshdata/2023data/celeba_HQ256/test')
    parser.add_argument('--testload', default=True, type=str2bool, help='whether test data is needed to validate the model')
    
    #------ parameter of train AE ------
    parser.add_argument('--epochs', type=int, default=1000, help='The whole Epochs for AE to train')#260
    parser.add_argument('--batch_size', type=int, default=256, help=' batch size of AE training')
    parser.add_argument('--dim_z', type=int, default=1200,help='dimension of feature in hidden space')
    parser.add_argument('--dim_c', type=int, default=3,help='input image number of channels')
    parser.add_argument('--dim_f', type=int, default=32,help='number of features in first layer of AE') ##32/64
    parser.add_argument('--lr', type=float, default=3e-5,help='learning rate of AE training')##  5e-4 1e-5
    parser.add_argument('--recnums', type=int, default=100,help='vistual numbers of reconstructed images')##
    parser.add_argument('--encoder_param', default=True, type=str2bool, help='Freeze parameter if setup False')
    parser.add_argument('--p', type=int, default=1,help='AE regular L1 or L2 loss')
    parser.add_argument('--wd', type=float, default=0.0,help='if wd>0 and lpips=0, runing mse + regular')###1e-6  
    parser.add_argument('--lpips_weight', default=0.0, type=float, help='if wd=0 and lpips>0, runing mse + lpips')##0.06
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay of optimal Adam')
    parser.add_argument('--opt_batch_size', default=100, type=int, help=' batch size of extract features')
    
    #------ parameter of calculate OT ------
    parser.add_argument('--maxIter', type=int, default=500,help='max iters of train ot')
    parser.add_argument('--lr_ot', type=int, default=12e-1,help='learning rate of calculate the update step based on gradient')
    parser.add_argument('--bat_size_n', type=int, default=4000,help='Size of mini-batch of Monte-Carlo samples on device')
    parser.add_argument('--init_num_bat_n', type=int, default=40,help='Starting number of mini-batch of Monte-Carlo samples')
    parser.add_argument('--eps', type=int, default=4e-3,help='error of g_norm of train OT')
    parser.add_argument('--N', type=int, default=30,help='cyclic generation of random measures to obtain the height vector and measure')
    parser.add_argument('--randmeasure', default=False, type=str2bool, help='rand measure or equal measure')

    #------ parameter of train OT via neural network ------
    parser.add_argument('--netepochs', type=int, default=1000, help='The whole Epochs for AE to train')#260
    parser.add_argument('--netbatch_size', type=int, default=4096, help=' batch size of AE training')## 4096/2e-4/0.9
    parser.add_argument('--feature_dim', type=int, default=1200, help=' dimension of feature')
    parser.add_argument('--measure_dim', type=int, default=1, help=' dimension of measure embedding')
    parser.add_argument('--lr_net', type=float, default=2e-4, help='learning rate of AE training')##  
    parser.add_argument('--weight', default=0.0, type=float, help='weight decay of optimal Adam') 
    parser.add_argument('--train_ratio', type=float, default=0.9, help='training sample ratio') 
    parser.add_argument('--result_root_path', type=str, default='./CelebAHQ1/', help='root path to save results')
    parser.add_argument('--hmodel', type=str, default='./CelebAHQ1/h_models/Epoch_979_models.pth', help='root path to h models')
    parser.add_argument('--nums_batch', type=int, default=30,help='The number of batches of h is obtained by pre-training the model')
    parser.add_argument('--train', default=True, type=str2bool, help='train model')
    parser.add_argument('--test_trainsample', default=True, type=str2bool, help='test h via pre-train model')
    parser.add_argument('--alter_measture', default=True, type=str2bool, help='achieving h via alter measure')
    parser.add_argument('--alter_measture_nums', default=10, type=float, help='achieving h via alter measure')
    parser.add_argument('--cons_samples', default=False, type=str2bool, help='given one class measure')
    parser.add_argument('--one_attribute', default=False, type=str2bool, help='given one attribute measure')
    parser.add_argument('--two_attribute', default=False, type=str2bool, help='given two attribute measure')
    
    #------ generated images -------------
    parser.add_argument('--max_gen_samples', type=int, default=1000,help='max number of generated samples')##
    parser.add_argument('--num_m', type=int, default=200,help='batch_size number of generated samples')##
    parser.add_argument('--gennums', type=int, default=100,help='vistual numbers of generated samples')##
    parser.add_argument('--image_size', type=int, default=256,help='image size of the generated samples')#MNSIT/fashion:28/cifar:32/celebA:64
    parser.add_argument('--angle_thresh', type=float, default=0.75,help='the threshold of the angle between two samples')
    parser.add_argument('--top_k', type=int, default=20, help='the nearest k samples around current sample')
    parser.add_argument('--dissim', type=float, default=0.05,help='the dissimilarity between the first generate new feature and the second generate new feature')
    
    #######################################FID and IS
    #parser.add_argument('--result_root_path', type=str, default='./celeba/', help='root path to save results')
    parser.add_argument("--data_generate", help='path to training set directory', type=str, metavar="", dest="data_generate", default='./CelebAHQ1/gen_imgs')
    #parser.add_argument("--data_generate", help='path to training set directory', type=str, metavar="", dest="data_generate", default='./rec_imgs/img_ema')  
    parser.add_argument("--fid_cache", help='path to testing set directory', type=str, metavar="", dest= "fid_cache",default='./stats/celeba.train.npz')
    
    #######################################dataset
    parser.add_argument('--dataset', type=str, default='FFHQ', help='reader dataset type')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--mdb', default=False, type=str2bool, help='train dataset is mdb or images')
    
    return parser.parse_args()

if __name__ == "__main__":
    # experiment setting
    args = get_args()
    if args.actions is None:
        actions = ['train_ae', 'refine_ae', 'extract_feature', 'train_ot', 'train_net_ot', 'generate_feature', 'decode_feature', 'calculate_fid']
    else:
        actions = args.actions

    # prepare the training arguments
    RESUME = True #toggles of whether to resume training
        
    subfolders = ['ae_models','rec_imgs','gen_imgs','gen_img_pairs','h_models']
    for i in range(len(subfolders)):
        fpath = os.path.join(args.result_root_path,subfolders[i])
        if not os.path.exists(fpath):
            os.makedirs(fpath)
    ae_model_path = os.path.join(args.result_root_path, 'ae_models')   
    ae_feature_path = os.path.join(args.result_root_path, 'ae_features.pt')  
    h_model_path = os.path.join(args.result_root_path,'h_models')    
    #Start training and/or generating
    for action in actions:
        if args.mdb:  
            dataset, testset = get_dataset(args)
        else:        
            img_transform = transforms.Compose([#transforms.Resize(args.image_size),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomVerticalFlip(),
                #transforms.RandomRotation(5),
                transforms.ToTensor(),
            ])
            test_transform = transforms.Compose([#transforms.Resize(args.image_size),
                transforms.ToTensor(),
            ])
            dataset = P_loader(root=args.data_root_train,transform=img_transform)
            testset = P_loader(root=args.data_root_test,transform=test_transform)
            
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        #print(len(dataset))    

        #########################################################################################################
        #model = autoencoder(args.dim_z, args.dim_c, args.dim_f, img_size_64=False,img_size_32=False).cuda()
        model = autoencoder(args.dim_z, args.dim_c, args.dim_f).cuda()
        model_net = DualInputNet(args.feature_dim, args.measure_dim).cuda()
        
        ''' train AE model '''
        if action == 'train_ae':
            train_ae(model,dataloader,testloader,args,resume=RESUME)
        else:
            model_load_path = ae_model_path
            if (not os.path.exists(ae_model_path)) or len(os.listdir(ae_model_path)) == 0:
                model_load_path = os.path.join(args.result_root_path,'ae_models')
            for file in os.listdir(model_load_path):
                if fnmatch.fnmatch(file, 'Epoch_*_sim_autoencoder*.pth'):
                    model.load_state_dict(torch.load(os.path.join(model_load_path, file)))
        
        ''' fine-tuning refine train AE model '''
        if action == 'refine_ae':
            fine_tuning_ae(model,dataloader,testloader,ae_model_path,args,resume=RESUME)
        

        ''' extract_feature of AE model '''
        if action == 'extract_feature':
            dataset = P_loader(root=args.data_root_train,transform=test_transform)
            dataloader_stable = DataLoader(dataset, batch_size=args.opt_batch_size, shuffle=False, drop_last=True, num_workers=4)
            data_len = len(dataset)
            extract_feature_ae(model,dataloader_stable,ae_model_path,args,data_len)
            args.feature_name = os.path.join(args.result_root_path,'ae_features.pt')

        ####  alter target measure
        fileName = './CelebAHQ1/CelebAHQ3.csv'
        data = pd.read_csv(fileName)
        if args.cons_samples:
            target_sample = torch.load(ae_feature_path)
            part = torch.div(torch.ones(args.alter_measture_nums), args.alter_measture_nums) 
            rest = torch.zeros(target_sample.size(0) - part.size(0))
            alter_target_measure = torch.cat((part,rest), dim=0)
            idxs = []
        elif args.one_attribute:
            column_data = data['Gender']
            tensor_data = torch.tensor(column_data.values)
            smile_data = data['Expression']
            smiling_data = torch.tensor(smile_data.values)
            race_data = data['beauty']
            racing_data = torch.tensor(race_data.values)
            greater_than_zero_idx = torch.nonzero(tensor_data > 0).squeeze() 
            less_than_zero_idx = torch.nonzero(tensor_data < 0).squeeze()
            '''greater_than_zero_idx = torch.nonzero( (tensor_data >= 15)&(tensor_data <= 24)& 
            (smiling_data > 0)&(racing_data == 1) ).squeeze() 
            less_than_zero_idx = torch.nonzero( torch.logical_or(tensor_data < 15, tensor_data > 24) | 
            (smiling_data < 0) | (racing_data != 1) ).squeeze()'''
            '''greater_than_zero_idx = torch.nonzero( (tensor_data < 0) & 
            (smiling_data > 0) & (racing_data >= 70) ).squeeze() 
            less_than_zero_idx = torch.nonzero( (tensor_data > 0) | 
            (smiling_data < 0) | (racing_data < 70 ) ).squeeze()'''
            greater_length = greater_than_zero_idx.size(0)
            less_length = less_than_zero_idx.size(0)
            print(greater_length, less_length)
            proportions = torch.tensor([greater_length, less_length])
            weights = 1 / proportions
            normalized_weights = weights / torch.sum(weights)
            print(normalized_weights)  ## gender/Age-normalized_weights//Eyeglasses-0.78/0.22//Expression-0.12/0.88 
 
            greater = torch.ones(greater_length)
            #greater = greater/torch.sum(greater)  
            greater = normalized_weights[0]*greater/torch.sum(greater)  

            less = torch.ones(less_length)
            #less = 0.12*less/torch.sum(less)
            less = normalized_weights[1]*less/torch.sum(less)
            
            alter_target_measure = torch.cat((greater,less), dim=0)
            idxs = [greater_than_zero_idx, less_than_zero_idx]
            print(torch.sum(alter_target_measure))
        elif args.two_attribute:
            gender_data = data['Gender']
            attr_data = data['Race']
            gender_data = torch.tensor(gender_data.values)
            attr_data = torch.tensor(attr_data.values)
            male_attr_idx = torch.nonzero( torch.logical_and(gender_data > 0, attr_data ==3 ) ).squeeze() 
            female_attr_idx = torch.nonzero( torch.logical_and(gender_data < 0, attr_data ==3) ).squeeze()
            male_attr = male_attr_idx.size(0)
            female_attr = female_attr_idx.size(0)
            male_Non_attr_idx = torch.nonzero( torch.logical_and(gender_data > 0, attr_data !=3) ).squeeze() 
            female_Non_attr_idx = torch.nonzero( torch.logical_and(gender_data < 0, attr_data !=3) ).squeeze()
            male_Non_attr = male_Non_attr_idx.size(0)
            female_Non_attr = female_Non_attr_idx.size(0)
            print('female&attr=',female_attr,'male&attr=',male_attr, 
                 'male&Non-attr=',male_Non_attr,'female&Non-attr=',female_Non_attr)
            proportions = torch.tensor([female_attr,male_attr,male_Non_attr,female_Non_attr])
            weights = 1 / proportions
            normalized_weights = weights / torch.sum(weights)
            print(normalized_weights) 
            #gender&race--h_pred2(0.56,0.35,0.03,0.06)/h_pred3(0.565,0.35,0.03,0.055)
            #gender&glass--(0.635,0.095,0.085,0.185)   #gender&smiling--(0.05,0.12,0.23,0.6)
            
            female_attr_measure = torch.ones(female_attr)
            #female_attr_measure = normalized_weights[0]*female_attr_measure/torch.sum(female_attr_measure)
            female_attr_measure = 0.64*female_attr_measure/torch.sum(female_attr_measure)

            male_attr_measure = torch.ones(male_attr)
            #male_attr_measure = normalized_weights[1]*male_attr_measure/torch.sum(male_attr_measure)
            male_attr_measure = 0.25*male_attr_measure/torch.sum(male_attr_measure)

            male_Non_attr_measure = torch.ones(male_Non_attr)
            #male_Non_attr_measure = normalized_weights[2]*male_Non_attr_measure/torch.sum(male_Non_attr_measure)
            male_Non_attr_measure = 0.06*male_Non_attr_measure/torch.sum(male_Non_attr_measure)

            female_Non_attr_measure = torch.ones(female_Non_attr)
            #female_Non_attr_measure = normalized_weights[3]*female_Non_attr_measure/torch.sum(female_Non_attr_measure)
            female_Non_attr_measure = 0.05*female_Non_attr_measure/torch.sum(female_Non_attr_measure)
            
            alter_target_measure = torch.cat((female_attr_measure,male_attr_measure,male_Non_attr_measure,female_Non_attr_measure), dim=0)
            idxs = [female_attr_idx, male_attr_idx,  male_Non_attr_idx,  female_Non_attr_idx]
            print(torch.sum(alter_target_measure))
        
        else:
            gender_data = data['Gender']
            race_data = data['Race']
            age_data = data['Age']
            gender_data = torch.tensor(gender_data.values)
            race_data = torch.tensor(race_data.values)
            age_data = torch.tensor(age_data.values)
            male_black_young_idx = torch.nonzero( (gender_data >0)& (race_data ==1)&(age_data <45) ).squeeze()  
            male_black_nonyoung_idx = torch.nonzero( (gender_data >0)& (race_data ==1)&(age_data >=45) ).squeeze()
            male_nonblack_young_idx = torch.nonzero( (gender_data >0)& (race_data !=1)&(age_data <45) ).squeeze()  
            male_nonblack_nonyoung_idx = torch.nonzero( (gender_data >0)& (race_data !=1)&(age_data >=45) ).squeeze()  
            male_black_young = male_black_young_idx.size(0)
            male_black_nonyoung = male_black_nonyoung_idx.size(0)
            male_nonblack_young = male_nonblack_young_idx.size(0)
            male_nonblack_nonyoung = male_nonblack_nonyoung_idx.size(0)

            female_black_young_idx = torch.nonzero( (gender_data <0)& (race_data ==1)&(age_data <45) ).squeeze()  
            female_black_nonyoung_idx = torch.nonzero( (gender_data <0)& (race_data ==1)&(age_data >=45) ).squeeze()
            female_nonblack_young_idx = torch.nonzero( (gender_data <0)& (race_data !=1)&(age_data <45) ).squeeze()  
            female_nonblack_nonyoung_idx = torch.nonzero( (gender_data <0)& (race_data !=1)&(age_data >=45) ).squeeze()  
            female_black_young = female_black_young_idx.size(0)
            female_black_nonyoung = female_black_nonyoung_idx.size(0)
            female_nonblack_young = female_nonblack_young_idx.size(0)
            female_nonblack_nonyoung = female_nonblack_nonyoung_idx.size(0)
            print('male_black_young=',male_black_young,'male_black_nonyoung=',male_black_nonyoung,
            'male_nonblack_young=',male_nonblack_young,'male_nonblack_nonyoung=',male_nonblack_nonyoung,
            'female_black_young=',female_black_young,'female_black_nonyoung=',female_black_nonyoung,
            'female_nonblack_young=',female_nonblack_young,'female_nonblack_nonyoung=',female_nonblack_nonyoung)

            proportions = torch.tensor( [male_black_young, male_black_nonyoung, male_nonblack_young,male_nonblack_nonyoung,
                                    female_black_young, female_black_nonyoung, female_nonblack_young,female_nonblack_nonyoung] )
            weights = 1 / proportions
            normalized_weights = weights / torch.sum(weights)
            print(normalized_weights) 

            male_black_young_measure = torch.ones(male_black_young)
            male_B_Y_measure = 0.003*male_black_young_measure/torch.sum(male_black_young_measure)
            male_black_nonyoung_measure = torch.ones(male_black_nonyoung)
            male_B_NY_measure = 0.6001*male_black_nonyoung_measure/torch.sum(male_black_nonyoung_measure)
            male_nonblack_young_measure = torch.ones(male_nonblack_young)
            male_NB_Y_measure = 0.0004*male_nonblack_young_measure/torch.sum(male_nonblack_young_measure)
            male_nonblack_nonyoung_measure = torch.ones(male_nonblack_nonyoung)
            male_NB_NY_measure = 0.003*male_nonblack_nonyoung_measure/torch.sum(male_nonblack_nonyoung_measure)

            female_black_young_measure = torch.ones(female_black_young)
            female_B_Y_measure = 0.0025*female_black_young_measure/torch.sum(female_black_young_measure)
            female_black_nonyoung_measure = torch.ones(female_black_nonyoung)
            female_B_NY_measure = 0.14*female_black_nonyoung_measure/torch.sum(female_black_nonyoung_measure)
            female_nonblack_young_measure = torch.ones(female_nonblack_young)
            female_NB_Y_measure = 0.001*female_nonblack_young_measure/torch.sum(female_nonblack_young_measure)
            female_nonblack_nonyoung_measure = torch.ones(female_nonblack_nonyoung)
            female_NB_NY_measure = 0.25*female_nonblack_nonyoung_measure/torch.sum(female_nonblack_nonyoung_measure)

            alter_target_measure = torch.cat((male_B_Y_measure,male_B_NY_measure,male_NB_Y_measure,male_NB_NY_measure,
                                female_B_Y_measure,female_B_NY_measure,female_NB_Y_measure,female_NB_NY_measure), dim=0)
            idxs = [male_black_young_idx, male_black_nonyoung_idx, male_nonblack_young_idx, male_nonblack_nonyoung_idx,
            female_black_young_idx, female_black_nonyoung_idx, female_nonblack_young_idx, female_nonblack_nonyoung_idx]
            print(torch.sum(alter_target_measure))
             

        ''' train(compute) OT with OT solver '''
        if action == 'train_ot':
            start = time.time()
            compute_ot(ae_feature_path, alter_target_measure, idxs,  args, mode='train')
            end = time.time()
            print("Train_ot done at %.3f seconds." % (end - start))

        if action == 'train_net_ot':
            start = time.time()
            train_ot_model(h_model_path, model_net, alter_target_measure,idxs,  args)
            end = time.time()
            print("Train_network_ot done at %.3f seconds." % (end - start))

        if action == 'generate_feature': ##generated new latent feature after OT
            print('Generating features with OT solver...') 
            compute_ot(ae_feature_path, alter_target_measure,idxs,  args, mode='generate')
            torch.cuda.empty_cache()

        ''' decode new latent feature '''    
        if action == 'decode_feature':
            decode_feature(ae_model_path,model,args)
        
        if action == 'calculate_fid':
            data_generate = os.path.join(args.data_generate+'/')
            print(data_generate)
            imagelist = os.listdir(data_generate)
            img = np.array(Image.open(data_generate+imagelist[1]))
            img = img.transpose((1,2,0)).transpose((1,0,2))
            print(img.shape)
            images =  np.zeros([len(imagelist),img.shape[0],img.shape[1],img.shape[2]]) 
            for i in range(len(imagelist)):
                img = np.array(Image.open(data_generate+imagelist[i]))
                img = img.transpose((1,2,0)).transpose((1,0,2))
                images[i,:,:,:] = img
            print(images.shape)
            (IS, IS_std), FID= get_inception_and_fid_score(images, data_generate, args.fid_cache, num_images=None,use_torch=False, verbose=True)
            print("IS:%6.5f(%.5f), FID:%7.5f" % (IS, IS_std, FID))
             