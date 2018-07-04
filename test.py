import cv2
import numpy as np

from config import config
import network as net
import os
import torch
from torch.autograd import Variable
import glob

class tester:
    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available():
            self.use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.use_cuda = False
            torch.set_default_tensor_type('torch.FloatTensor')

        self.nz = config.nz

        self.resl = config.resl  # we start from 2^2 = 4
        self.lr = config.lr
        self.eps_drift = config.eps_drift
        self.smoothing = config.smoothing
        self.max_resl = config.max_resl
        self.trns_tick = config.trns_tick
        self.stab_tick = config.stab_tick
        self.TICK = config.TICK
        self.globalIter = 0
        self.globalTick = config.globalTick
        self.kimgs = 0
        self.stack = 0
        self.epoch = 0
        self.fadein = {'gen': None, 'dis': None}
        self.complete = {'gen': 0, 'dis': 0}
        self.phase = 'init'
        self.flag_flush_gen = False
        self.flag_flush_dis = False
        self.flag_add_noise = self.config.flag_add_noise
        self.flag_add_drift = self.config.flag_add_drift
        self.load = config.load
        self.batch = 4

        # network and cirterion
        self.D = net.Discriminator(config)

        if self.use_cuda:
            if config.n_gpu == 1:
                self.D = torch.nn.DataParallel(self.D).cuda(device=0)
            else:
                gpus = []
                for i in range(config.n_gpu):
                    gpus.append(i)
                self.D = torch.nn.DataParallel(self.D, device_ids=gpus).cuda()

        self.renew_everything()
        if self.load:
            self.load_snapshot('repo/model')
        self.renew_everything()

        if self.use_cuda:
            if config.n_gpu == 1:
                self.D.cuda(device=0)

        self.loader = loader()

    def renew_everything(self):
        # define tensors
        #self.x = torch.FloatTensor(self.loader.batchsize, 3, self.loader.imsize, self.loader.imsize)
        self.x = torch.FloatTensor(self.batch, 3, 128, 128)

        # enable cuda
        if self.use_cuda:
            self.x = self.x.cuda()

        # wrapping autograd Variable.
        self.x = Variable(self.x)

        # ship new model to cuda.
        if self.use_cuda:
            self.D = self.D.cuda()

    def test(self):
        for index in range(100):
            # update discriminator.
            #self.x.data = self.feed_interpolated_input(self.loader.get_batch())
            self.x.data = self.loader.load_batch_picture(self.batch)

            self.fx = self.D(self.x)
            self.fx_tilde = self.D(self.x_tilde.detach())

    def load_snapshot(self, path='repo/model'):
        snapshots_file = glob.glob(os.path.join(path, '*.pth.tar'))
        diss = [_file for _file in snapshots_file if 'dis_R' in _file]
        diss.sort()
        if len(diss) == 0:
            print('No model detected, training from skretch')
            return
        dis = diss[-1]
        print('Loading {} from disk'.format(dis))
        net = self.D
        snap = dis
        mode = 'd'
        _, R, T = snap.split('_') #gen Rx Txxx.pth.tar
        T, _, _ = T.split('.') #Txxxx pth tar
        resl = int(R[1:])
        for i in range(3, resl+1):
            net.module.grow_network(i)
            net.module.flush_network()
        checkpoint = torch.load(snap)
        #opt.load_state_dict(checkpoint['optimizer'])
        state_dict = {}
        for key, val in checkpoint['state_dict'].items():
            if mode == 'g':
                if not ("concat_block" in key or "fadein_block" in key):
                    state_dict[key] = val
                else:
                    print (key)
            if mode == 'd':
                if not ("concat_block" in key or "fadein_block" in key or "from_rgb_block" in key):
                    state_dict[key] = val
                else:
                    print (key)
        net.module.load_state_dict(state_dict, strict=False)

        self.resl = resl
        self.globalTick = int(T[1:])
        print('model load @ {}'.format(path))

class loader:
    def __init__(self):
        self.video = cv2.VideoCapture('videoplayback.mp4')

    def load_batch_picture(self, batch):
        batch_picture = []
        index = 0
        while (self.video.isOpened() and index < batch):
            ret, frame = self.video.read()

            # frame is image
            if ret == True:
                #cv2.imshow('frame', frame)
                print (frame.shape)
                batch_picture.append(frame)
                index += 1
            else:
                break
            if cv2.waitKey(3) & 0xFF == ord('q'):
                break

        return batch_picture

if __name__ == "__main__":
    test = tester(config)
    test.test()
