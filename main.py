#!/usr/bin/env python
import os
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from chainer import optimizers
from chainer import serializers

import cv2

image_size = 64
nbatch = 32
n_input = image_size * image_size
nz = 300

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
            #gen_log_sigma = L.Liner(500, n_input)
        )

    def __call__(self, x_data, y_data):
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)

        # q(z|x,y)
        rh1 = self.recog1(x)
        rh2 = self.recog2(rh1)
        recog_mean = self.recog_mean(rh2)
        recog_log_sigma = 0.5 * self.recog_log_sigma(rh2)

        eps = np.random.normal(0, 1, (x.data.shape[0], nz)).astype(np.float32)
        eps = chainer.Variable(eps)

        # z = mu + sigma + epsilon
        z = recog_mean + F.exp(recog_log_sigma) * eps

        gh1 = self.gen1(z)
        gh2 = self.gen2(gh1)
        gen_mean = self.gen_mean(gh2)
        output = F.sigmoid(gen_mean)
        loss = F.mean_squared_error(output, y)
        kld = -0.5 * F.sum(1 + recog_log_sigma - recog_mean**2 - F.exp(recog_log_sigma)) / (x_data.shape[0] * x_data.shape[1])
        return loss, kld, output


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


for epoch in xrange(500000):
    print "epoch:", epoch

    xmb = np.zeros((nbatch, image_size*image_size), dtype=np.float32)
    for j in range(nbatch):
        rnd = np.random.randint(2000)
        xmb[j,:] = dataset[rnd]

    x_data = xmb
    y_data = x_data
    recog_loss, kld_loss, output = vae(x_data, y_data)
    loss = recog_loss + kld_loss
    vae.zerograds()
    loss.backward()
    opt.update()

    for j in range(0, 3):
        img = output.data[j]
        img = img * 255
        img = img.reshape(image_size, image_size)
        img = img.astype(np.uint8)
        cv2.imshow("%d"%j, img)

    cv2.waitKey(1)

    if epoch % 300 == 0:
        for j in range(0, nbatch):
            img = output.data[j]
            img = img * 255
            img = img.reshape(image_size, image_size)
            img = img.astype(np.uint8)
            cv2.imwrite("out_images/%d_%d.jpg"%(epoch, j), img)
