from os.path import dirname, join, basename, isfile
# import  paddle
from tqdm import tqdm
from vgg import VGGLoss
from color_loss import Blur, ColorLoss
from models import SyncNet_color as SyncNet
from models import LNet, Wav2Lip_disc_qual
import audio
from boundary_loss import HDLoss
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils import data as data_utils
import numpy as np
from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list



parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model WITH the visual quality discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True, type=str)

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--syncnet_checkpoint_path', help='Load the pre-trained Expert discriminator', required=True, type=str)

parser.add_argument('--checkpoint_path', help='Resume generator from this checkpoint', default=None, type=str)
parser.add_argument('--disc_checkpoint_path', help='Resume quality disc from this checkpoint', default=None, type=str)

args = parser.parse_args()


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16

class Dataset(object):
    def __init__(self, split):
        self.all_videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])
# 获取视频窗口
    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                # 裁剪图片大小
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            # 行=16
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            # 随机选择一个视频id
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                continue
            # 随机选择帧
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue

            window = self.read_window(window_fnames)
            if window is None:
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                continue

            try:
                # 读取音频
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)
                # 提取完整mel-spectrogram
                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue
            # 分割 mel-spectrogram
            mel = self.crop_audio_window(orig_mel.copy(), img_name)
            
            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None: continue
            # winmask = []
            # for i in range(0, len(window)):
            #     winmask.append(lipseg.mask(window[i]))
            # # toarray
            # window = self.prepare_window(winmask)
            # y = window.copy()
            # window[:, :, window.shape[2]//2:] = 0.
            window = self.prepare_window(window)
            y = window.copy()
            window[:, :, window.shape[2] // 2:] = 0.


            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)

            x = torch.FloatTensor(x)
            # unsqueeze：升维
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            y = torch.FloatTensor(y)
            # x:输入的图片
            # indiv_mels: 每一张图片所对应语音的mel-spectrogram特征
            # mel: 所有帧对应的200ms的语音mel-spectrogram，用于SyncNet进行唇音同步损失的计算
            # y:真实的与语音对应的，唇音同步的图片
            return x, indiv_mels, mel, y
# Check the norm of gradients
def get_grad_norm(model):
    # from https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961/2
    norm_type = 2
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        total_norm = 0.0
    else:
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), 2.0).item()

    return total_norm

def clamp_grad_norm_(parameters):
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]), 2.0)
    clip_coef = hparams.disc_max_grad_norm / (total_norm + 1e-6)
    stretch_coef = hparams.disc_min_grad_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
            for p in parameters:
                        p.grad.detach().mul_(clip_coef.to(p.grad.device))
    elif stretch_coef > 1.0:
            for p in parameters:
                        p.grad.detach().mul_(stretch_coef.to(p.grad.device))

def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    # g = cv2.bilateralFilter(g, 9, 150, 150)
    refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])



logloss = nn.BCELoss()
# consine_loss为二元交叉熵loss
def cosine_loss(a, v, y):
    # 余弦相似度
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss



device = torch.device("cuda" if use_cuda else "cpu")
syncnet = SyncNet().to(device)
for p in syncnet.parameters():
    p.requires_grad = False

# 边缘loss
net = HDLoss().to(device)
recon_loss = nn.L1Loss()
vgg_l = VGGLoss(device=device)

# -------------------------自定义colorloss
def get_color_loss(g,gt):
    cl = ColorLoss()
    blur_rgb = Blur(3)
    g = g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1)
    gt = gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1)
    b, t, h, w ,c = g.shape
    g = g.reshape((b*t), c, h ,w)
    b, t, h, w, c = gt.shape
    gt = gt.reshape((b * t), c, h,w)
    g = torch.from_numpy(g)
    gt = torch.from_numpy(gt)
    g_c = blur_rgb(g).to(device)
    gt_c = blur_rgb(gt).to(device)
    closs=cl(g_c, gt_c).to(device)
    return closs
def get_style_loss(g,gt):
    g = gram(g)
    gt = gram(gt)
    g = torch.from_numpy(g)
    gt = torch.from_numpy(gt)
    styloss = style_loss(g ,gt)
    return styloss
def gram(x):
    x = x.detach().cpu().numpy().transpose(0, 2, 1, 3, 4)
    x = torch.from_numpy(x)
    b, t, c, h, w = x.shape
    x_tmp = x.reshape(((b*t), c, (h * w)))
    xx = x_tmp.detach().cpu().numpy().transpose(1,2,0 )
    xx = torch.from_numpy(xx)
    gram = np.dot(x_tmp, xx)
    return gram / (c * h * w)


def style_loss(fake,style):

    gram_loss = nn.L1Loss()(style, fake)
    return gram_loss

def get_sync_loss(mel, g):
    # (B, C, T, H, W)
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g)
    # audio_embedding, face_embedding
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)
def re(x):
    x = x.detach().cpu().numpy().transpose(0, 2, 1, 3, 4)
    x = torch.from_numpy(x)
    b, t, c, h, w = x.shape
    x = x.reshape(((b * t), c, h ,w))
    return x
def train(device, model, disc, train_data_loader, test_data_loader, optimizer, disc_optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch
    resumed_step = global_step

    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_sync_loss, running_l1_loss, disc_loss, running_perceptual_loss ,running_color_loss,running_style_loss= 0. ,0., 0., 0., 0.,0.
        running_boundary_loss, running_vgg_loss = 0, 0
        running_disc_real_loss, running_disc_fake_loss = 0., 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, indiv_mels, mel, gt) in prog_bar:
            disc.train()
            model.train()

            x = x.to(device)    #16.6.5.96.96
            mel = mel.to(device)  # 16.1.80.16
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)  #16.3.5.96.96 (B, C, T, H, W)

            ### Train generator now. Remove ALL grads.
            # 梯度清零
            optimizer.zero_grad()
            disc_optimizer.zero_grad()
            # 生成结果
            g = model(indiv_mels, x) #B.6.5.96.96

            if hparams.syncnet_wt > 0.:
                # 从预训练的expert 模型中获得唇音同步的损失
                sync_loss = get_sync_loss(mel, g)
            else:
                sync_loss = 0.

            if hparams.disc_wt > 0.:
                # 判别器的感知损失
                perceptual_loss = disc.perceptual_forward(g)
            else:
                perceptual_loss = 0.
# 获得loss
            input = re(g).to(device)
            target= re(gt).to(device)
            vgg_loss = vgg_l(input, target)
            # boundary_loss = net(input, target)
            l1loss = recon_loss(g, gt)
            color_loss =  get_color_loss(g,gt)*0.001
            style_loss = get_style_loss(g,gt)




            loss = hparams.syncnet_wt * sync_loss + hparams.disc_wt * perceptual_loss + \
                                    (1. - hparams.syncnet_wt - hparams.disc_wt -0.06) * l1loss + \
                                     0.01 * color_loss +0.05* vgg_loss
                   # 0.05 * boundary_loss \
            #反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            ### Remove all gradients before Training disc
            disc_optimizer.zero_grad()

            pred = disc(gt)
            #二元交叉熵loss ：binary_cross_entropy(input,target)
            #yi为1，pred为face，所以real->0，最小化real_loss
            disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(device))
            disc_real_loss.backward()
            # yi为0，pred为false，最小化false_loss
            pred = disc(g.detach())
            disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))
            disc_fake_loss.backward()

            if hparams.disc_max_grad_norm and hparams.disc_min_grad_norm:
                # torch.nn.utils.clip_grad_norm_(disc.parameters(), hparams.disc_max_grad_norm)
                clamp_grad_norm_(disc.parameters())
            disc_optimizer.step()
            disc_grad_norm = get_grad_norm(disc)
            running_disc_real_loss += disc_real_loss.item()
            running_disc_fake_loss += disc_fake_loss.item()

            if global_step % checkpoint_interval == 0:
                # x：视频，g：生成的，gt：真实的（参考）
                save_sample_images(x, g, gt, global_step, checkpoint_dir)

            # Logs
            global_step += 1
            cur_session_steps = global_step - resumed_step
            # running_boundary_loss += boundary_loss.item()
            running_vgg_loss += vgg_loss.item()
            running_color_loss += color_loss.item()
            running_style_loss += style_loss.item()
            running_l1_loss += l1loss.item()
            if hparams.syncnet_wt > 0.:
                running_sync_loss += sync_loss.item()
            else:
                running_sync_loss += 0.

            if hparams.disc_wt > 0.:
                running_perceptual_loss += perceptual_loss.item()
            else:
                running_perceptual_loss += 0.

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)
                save_checkpoint(disc, disc_optimizer, global_step, checkpoint_dir, global_epoch, prefix='disc_')


            if global_step % hparams.eval_interval == 0:
                # 测试一下

                with torch.no_grad():
                    average_sync_loss = eval_model(test_data_loader, global_step, device, model, disc)
                    ave_sync = []
                    ave_sync.append(average_sync_loss)
                    with open(join(args.checkpoint_dir, "train.log"), "a") as train_log:
                        train_log.write(
                            '###### Now at global epoch {} and global step{:09d} synv:{}#####\n'.format(global_epoch,
                                                                                                        global_step,
                                                                                                        average_sync_loss))
                    if average_sync_loss < 0.75:
                        hparams.set_hparam('syncnet_wt', 0.03)

            prog_bar.set_description('ph:{}, L1: {}, Sync: {}, Percep: {} | F: {}, T: {} \n color: {}, vgg:{} norm:{}'.format(global_epoch,
                                                                                        running_l1_loss / (step + 1),
                                                                                        running_sync_loss / (step + 1),
                                                                                        running_perceptual_loss / (step + 1),
                                                                                        running_disc_fake_loss / (step + 1),
                                                                                        running_disc_real_loss / (step + 1),
                                                                                        running_color_loss / (step + 1),
                                                                                        # running_style_loss / (step + 1),
                                                                                        running_vgg_loss/(step + 1),

                                                                                        disc_grad_norm))

        global_epoch += 1



def eval_model(test_data_loader, global_step, device, model, disc):
    eval_steps = 300
    print('Evaluating for {} steps'.format(eval_steps))
    running_boundary_loss, running_vgg_loss = 0, 0
    running_sync_loss,running_color_loss, running_l1_loss, running_disc_real_loss, running_disc_fake_loss, running_perceptual_loss, running_style_loss = [], [], [], [], [],[],[]
    while 1:
        for step, (x, indiv_mels, mel, gt) in enumerate((test_data_loader)):
            model.eval()
            disc.eval()

            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)

            pred = disc(gt)
            disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(device))

            g = model(indiv_mels, x)
            pred = disc(g)
            disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))

            running_disc_real_loss.append(disc_real_loss.item())
            running_disc_fake_loss.append(disc_fake_loss.item())
            # 获得loss
            input = re(g).to(device)
            target = re(gt).to(device)
            vgg_loss = vgg_l(input, target)
            boundary_loss = net(g, gt)
            sync_loss = get_sync_loss(mel, g)
            style_loss = get_style_loss(g,gt)
            if hparams.disc_wt > 0.:
                perceptual_loss = disc.perceptual_forward(g)
            else:
                perceptual_loss = 0.

            l1loss = recon_loss(g, gt)
            color_loss = get_color_loss(g, gt) *0.001
            loss = hparams.syncnet_wt * sync_loss + hparams.disc_wt * perceptual_loss + \
                                    (1. - hparams.syncnet_wt - hparams.disc_wt -0.4) * l1loss + \
                                     0.01 * color_loss +0.05* vgg_loss


            running_vgg_loss.append(vgg_loss)
            running_boundary_loss.append(boundary_loss)
            running_vgg_loss += vgg_loss.item()
            running_l1_loss.append(l1loss.item())
            running_sync_loss.append(sync_loss.item())
            running_color_loss.append(color_loss.item())
            running_style_loss.append(style_loss.item())
            if hparams.disc_wt > 0.:
                running_perceptual_loss.append(perceptual_loss.item())
            else:
                running_perceptual_loss.append(0.)

            if step > eval_steps: break

        print('L1: {}, Sync: {}, Percep: {} | Fake: {}, Real: {} ,Color:{}, vgg{}: '.format(sum(running_l1_loss) / len(running_l1_loss),
                                                            sum(running_sync_loss) / len(running_sync_loss),
                                                            sum(running_perceptual_loss) / len(running_perceptual_loss),
                                                            sum(running_disc_fake_loss) / len(running_disc_fake_loss),
                                                            sum(running_disc_real_loss) / len(running_disc_real_loss),
                                                            sum(running_color_loss) / len(running_color_loss),
                                                            sum(running_vgg_loss) / len(running_vgg_loss),
                                                            # sum(running_boundary_loss) / len(running_boundary_loss),
                                                             ))

        return sum(running_sync_loss) / len(running_sync_loss)


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = join(
        checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model

if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir

    # Dataset and Dataloader setup
    train_dataset = Dataset('train')
    test_dataset = Dataset('val')

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.batch_size, shuffle=True,
        num_workers=0)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.batch_size,
        num_workers=0)

    device = torch.device("cuda" if use_cuda else "cpu")

     # Model
    model = LNet().to(device)
    disc = Wav2Lip_disc_qual().to(device)

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('total DISC trainable params {}'.format(sum(p.numel() for p in disc.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam([p for p in disc.parameters() if p.requires_grad],
                           lr=hparams.disc_initial_learning_rate, betas=(0.5, 0.999))

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)

    if args.disc_checkpoint_path is not None:
        load_checkpoint(args.disc_checkpoint_path, disc, disc_optimizer, 
                                reset_optimizer=False, overwrite_global_states=False)
        
    load_checkpoint(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True, 
                                overwrite_global_states=False)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # Train!
    train(device, model, disc, train_data_loader, test_data_loader, optimizer, disc_optimizer,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=hparams.nepochs)
