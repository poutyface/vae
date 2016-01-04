#!/usr/bin/env python
import os
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import Chain
from chainer import optimizers
from chainer import serializers

import cv2

image_size = 96
nbatch = 32
n_input = image_size * image_size
nz = 30
nc = 1
ndf = 64


class VAE(Chain):
    def __init__(self):
        super(VAE, self).__init__(
            recog1 = L.Linear(n_input, 500),
            recog2 = L.Linear(500, 500),
            recog_mean = L.Linear(500, nz),
            recog_log_sigma = L.Linear(500, nz),
            gen1 = L.Linear(nz, 500),
            gen2 = L.Linear(500, 500),
            gen_mean = L.Linear(500, n_input),

            #bn1 = L.BatchNormalization(ndf*2),
            #bn2 = L.BatchNormalization(ndf*4),
            #bn3 = L.BatchNormalization(ndf*8),
            #gen_log_sigma = L.Liner(500, n_input)
        )

    def __call__(self, x_data, y_data):
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)

        # q(z|x,y)
        rh1 = F.relu(self.recog1(x))
        rh2 = F.relu(self.recog2(rh1))
        recog_mean = self.recog_mean(rh2)
        #recog_log_sigma = 0.5 * self.recog_log_sigma(rh2)
        recog_log_sigma = self.recog_log_sigma(rh2)

        eps = np.random.normal(0, 1, (x.data.shape[0], nz)).astype(np.float32)
        eps = chainer.Variable(eps)

        # z = mu + sigma + epsilon
        z = recog_mean + F.exp(0.5 * recog_log_sigma) * eps
        #z = recog_mean + F.exp(recog_log_sigma) * eps

        gh1 = F.relu(self.gen1(z))
        gh2 = F.relu(self.gen2(gh1))
        gen_mean = self.gen_mean(gh2)
        output = F.sigmoid(gen_mean)
        loss = F.mean_squared_error(output, y)
        kld = -0.5 * F.sum(1 + recog_log_sigma - recog_mean**2 - F.exp(recog_log_sigma)) / (x_data.shape[0] * x_data.shape[1])
        return loss, kld, output



class Disc(Chain):
    def __init__(self):
        super(Disc, self).__init__(
            bn1 = L.BatchNormalization(ndf*2),
            bn2 = L.BatchNormalization(ndf*4),
            bn3 = L.BatchNormalization(ndf*8),
            c1 = L.Convolution2D(nc, ndf, ksize=4, stride=2, pad=1),
            c2 = L.Convolution2D(ndf, ndf*2, ksize=4, stride=2, pad=1),
            c3 = L.Convolution2D(ndf*2, ndf*4, ksize=4, stride=2, pad=1),
            c4 = L.Convolution2D(ndf*4, ndf*8, ksize=4, stride=2, pad=1),
            l1 = L.Linear(ndf*8*6*6, 1)
        )

    def __call__(self, x, test=False):
        h1 = F.leaky_relu(self.c1(x))
        h2 = F.leaky_relu(self.bn1(self.c2(h1), test=test))
        h3 = F.leaky_relu(self.bn2(self.c3(h2), test=test))
        h4 = F.leaky_relu(self.bn3(self.c4(h3), test=test))
        #h2 = F.leaky_relu(self.c2(h1))
        #h3 = F.leaky_relu(self.c3(h2))
        #h4 = F.leaky_relu(self.c4(h3))
        #h5 = F.average_pooling_2d(h4, 4)
        #h5 = self.l1(h4)
        h5 = self.l1(h4)

        print x.data.shape
        print h1.data.shape
        print h2.data.shape
        print h3.data.shape
        print h4.data.shape
        print h5.data.shape
        #print h6.data.shape
        return h5


image_path = "./lfwcrop_grey/faces"

fs = os.listdir(image_path)
print len(fs)
dataset = []
for fn in fs:
    # read as grey
    img = cv2.imread('%s/%s'%(image_path, fn), 0)
    img = cv2.resize(img, (image_size,image_size))
    img = img.astype(np.float32)
    img = img / 255
    img = img.reshape(image_size*image_size)
    dataset.append(img)


vae = VAE()
opt = optimizers.Adam(alpha=0.0002, beta1=0.5)
opt.setup(vae)

disc = Disc()
d_opt = optimizers.Adam(alpha=0.0002, beta1=0.5)
d_opt.setup(disc)


for epoch in xrange(500000):
    print "epoch:", epoch

    xmb = np.zeros((nbatch, image_size*image_size), dtype=np.float32)
    for j in range(nbatch):
        rnd = np.random.randint(5000)
        xmb[j,:] = dataset[rnd]

    x_data = xmb
    y_data = x_data
    # VAE
    recog_loss, kld_loss, output = vae(x_data, y_data)


    # y1 = disc(F.reshape(output, (nbatch, nc, image_size, image_size)))
    # l_gen = F.sigmoid_cross_entropy(y1, chainer.Variable(np.ones((nbatch, 1), dtype=np.int32)))
    # l_fake = F.sigmoid_cross_entropy(y1, chainer.Variable(np.zeros((nbatch, 1), dtype=np.int32)))
    #
    # x2 = F.reshape(chainer.Variable(x_data), (nbatch, nc, image_size, image_size))
    # y2 = disc(x2)
    # l_real = F.sigmoid_cross_entropy(y2, chainer.Variable(np.ones((nbatch, 1), dtype=np.int32)))
    # l_dis = l_fake + l_real
    #
    #
    # disc.zerograds()
    # l_dis.backward()
    # d_opt.update()

#    loss = recog_loss + kld_loss + l_gen
    loss = recog_loss + kld_loss
    print loss.data
    vae.zerograds()
    loss.backward()
    opt.update()

    if epoch % 100 == 0:
        for j in range(0, 3):
            img = output.data[j]
            img = img * 255
            img = img.reshape(image_size, image_size)
            img = img.astype(np.uint8)
            cv2.imshow("%d"%j, img)

        cv2.waitKey(1)

    if epoch % 1000 == 0:
        for j in range(0, nbatch):
            img = output.data[j]
            img = img * 255
            img = img.reshape(image_size, image_size)
            img = img.astype(np.uint8)
            cv2.imwrite("out_images/%d_%d.jpg"%(epoch, j), img)

        serializers.save_hdf5("out_models/model_%d.h5"%(epoch), vae)
