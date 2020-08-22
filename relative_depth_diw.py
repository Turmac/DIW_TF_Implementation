import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import math


batch_size = 4
input_height = 240
input_width = 320


def inception(x, input_dim, output_dim, config, layer_name):
    with tf.variable_scope(layer_name):
        # 1st layer
        h1 = tf.layers.conv2d(inputs=x, filters=output_dim, kernel_size=config[0][0], padding='SAME')
        h1 = tf.layers.batch_normalization(inputs=h1, momentum=0.1, epsilon=1e-5)
        h1 = tf.nn.relu(h1)

        # 2nd layer
        h2 = tf.layers.conv2d(inputs=x, filters=config[1][1], kernel_size=1, padding='SAME')
        h2 = tf.layers.batch_normalization(inputs=h2, momentum=0.1, epsilon=1e-5)
        h2 = tf.nn.relu(h2)
        h2 = tf.layers.conv2d(inputs=h2, filters=output_dim, kernel_size=config[1][0], padding='SAME')
        h2 = tf.layers.batch_normalization(inputs=h2, momentum=0.1, epsilon=1e-5)
        h2 = tf.nn.relu(h2)

        # 3rd layer
        h3 = tf.layers.conv2d(inputs=x, filters=config[2][1], kernel_size=1, padding='SAME')
        h3 = tf.layers.batch_normalization(inputs=h3, momentum=0.1, epsilon=1e-5)
        h3 = tf.nn.relu(h3)
        h3 = tf.layers.conv2d(inputs=h3, filters=output_dim, kernel_size=config[2][0], padding='SAME')
        h3 = tf.layers.batch_normalization(inputs=h3, momentum=0.1, epsilon=1e-5)
        h3 = tf.nn.relu(h3)

        # 4th layer
        h4 = tf.layers.conv2d(inputs=x, filters=config[3][1], kernel_size=1, padding='SAME')
        h4 = tf.layers.batch_normalization(inputs=h4, momentum=0.1, epsilon=1e-5)
        h4 = tf.nn.relu(h4)
        h4 = tf.layers.conv2d(inputs=h4, filters=output_dim, kernel_size=config[3][0], padding='SAME')
        h4 = tf.layers.batch_normalization(inputs=h4, momentum=0.1, epsilon=1e-5)
        h4 = tf.nn.relu(h4)

        return tf.concat([h1, h2, h3, h4], axis=3)


def Channel1(x):
    with tf.variable_scope('Channel1'):
        h1 = inception(x, 256, 64, [[1], [3, 32], [5, 32], [7, 32]], 'E11')
        h1 = inception(h1, 256, 64, [[1], [3, 32], [5, 32], [7, 32]], 'E12')

        h2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
        h2 = inception(h2, 256, 64, [[1], [3, 32], [5, 32], [7, 32]], 'E13')
        h2 = inception(h2, 256, 64, [[1], [3, 32], [5, 32], [7, 32]], 'E14')
        h2 = inception(h2, 256, 64, [[1], [3, 32], [5, 32], [7, 32]], 'E15')
        h2 = tf.keras.layers.UpSampling2D(size=(2, 2))(h2)

        return h1 + h2


def Channel2(x):
    with tf.variable_scope('Channel2'):
        h1 = inception(x, 256, 64, [[1], [3, 32], [5, 32], [7, 32]], 'E21')
        h1 = inception(h1, 256, 64, [[1], [3, 64], [7, 64], [11, 64]], 'F21')

        h2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
        h2 = inception(h2, 256, 64, [[1], [3, 32], [5, 32], [7, 32]], 'E22')
        h2 = inception(h2, 256, 64, [[1], [3, 32], [5, 32], [7, 32]], 'E23')
        h2 = Channel1(h2)
        h2 = inception(h2, 256, 64, [[1], [3, 32], [5, 32], [7, 32]], 'E24')
        h2 = inception(h2, 256, 64, [[1, 64], [3, 64], [7, 64], [11, 64]], 'F22')
        h2 = tf.keras.layers.UpSampling2D(size=(2, 2))(h2)

        return h1 + h2


def Channel3(x):
    with tf.variable_scope('Channel3'):
        h1 = inception(x, 128, 32, [[1], [3, 32], [5, 32], [7, 32]], 'B31')
        h1 = inception(h1, 128, 32, [[1], [3, 64], [7, 64], [11, 64]], 'C31')

        h2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
        h2 = inception(h2, 128, 32, [[1], [3, 32], [5, 32], [7, 32]], 'B32')
        h2 = inception(h2, 128, 64, [[1], [3, 32], [5, 32], [7, 32]], 'D31')
        h2 = Channel2(h2)
        h2 = inception(h2, 256, 64, [[1], [3, 32], [5, 32], [7, 32]], 'E31')
        h2 = inception(h2, 256, 32, [[1], [3, 32], [5, 32], [7, 32]], 'G31')
        h2 = tf.keras.layers.UpSampling2D(size=(2, 2))(h2)

        return h1 + h2


def Channel4(x):
    with tf.variable_scope('Channel4'):
        h1 = inception(x, 128, 16, [[1], [3, 64], [7, 64], [11, 64]], 'A41')

        h2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
        h2 = inception(h2, 128, 32, [[1], [3, 32], [5, 32], [7, 32]], 'B41')
        h2 = inception(h2, 128, 32, [[1], [3, 32], [5, 32], [7, 32]], 'B42')
        h2 = Channel3(h2)
        h2 = inception(h2, 128, 32, [[1], [3, 64], [5, 64], [7, 64]], 'B43')
        h2 = inception(h2, 128, 16, [[1], [3, 32], [7, 32], [11, 32]], 'A42')
        h2 = tf.keras.layers.UpSampling2D(size=(2, 2))(h2)

        return h1 + h2


def hourglass(x):
    with tf.variable_scope('hourglass'):
        x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=7, padding='SAME')
        x = tf.contrib.layers.batch_norm(inputs=x, decay=0.9, epsilon=1e-5)
        x = tf.nn.relu(x)
        x = Channel4(x)
        last_feature_map = x
        x = tf.layers.conv2d(inputs=x, filters=1, kernel_size=3, padding='SAME')
        return x, last_feature_map


def feed_forward(x):
    x, feature_map = hourglass(x)
    return x, feature_map


def feed_forward_2(x):
    with tf.variable_scope('lf_depth'):
        W = tf.get_variable('weights', [3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        #B = tf.get_variable('bias', [64], initializer=tf.zeros_initializer())
        #x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=3, padding='SAME')
        x = tf.nn.conv2d(input=x, filter=W, strides=[1,1,1,1], padding='SAME')
        #x = x + B
        return x


def relative_loss(za, zb, relation):
    """
    Returns:
        1-D Tensor - loss
    """
    mask = tf.abs(relation)
    return mask*tf.log(1+tf.exp(-relation*(za-zb)))+(1-mask)*(za-zb)*(za-zb)


def relative_loss_criterion(batch_output, target):
    """
    Args:
        batch_output: 4-D tensor, [batch, height, width, channel]
        target: 3-D tensor, relative depth data, [batch, num_pairs, 5]
                where the 5 number is - yA, xA, yB, xB, label (x: width, y: height)
    """
    output = tf.Variable(0.0)

    for batch_idx in range(batch_output.shape[0].value):
        yA_tensor = tf.slice(target, [batch_idx, 0, 0], [1, -1, 1])
        xA_tensor = tf.slice(target, [batch_idx, 0, 1], [1, -1, 1])
        yB_tensor = tf.slice(target, [batch_idx, 0, 2], [1, -1, 1])
        xB_tensor = tf.slice(target, [batch_idx, 0, 3], [1, -1, 1])
        
        depth = tf.squeeze(batch_output[batch_idx], -1)
        coord_A = tf.stack([yA_tensor, xA_tensor], -1)
        coord_B = tf.stack([yB_tensor, xB_tensor], -1)
        zA_tensor = tf.gather_nd(depth, coord_A)
        zB_tensor = tf.gather_nd(depth, coord_B)

        ground_truth = tf.slice(target, [batch_idx, 0, 4], [1, -1, 1])
        ground_truth = tf.to_float(ground_truth)

        output += tf.reduce_sum(relative_loss(zA_tensor, zB_tensor, ground_truth))

    num_pairs = tf.to_float(tf.shape(target)[1])
    return output/num_pairs


X = tf.placeholder(tf.float32, shape=(batch_size, input_height, input_width, 3))
Y = tf.placeholder(tf.int32, shape=(batch_size, None, 5))
y_, fm = feed_forward(X)

tf.summary.image('input', X, 4)
tf.summary.image('output', y_, 4)

lfd = feed_forward_2(fm)

ymin = tf.reduce_min(y_)
ymax = tf.reduce_max(y_ - ymin)
depth = (y_ - ymin)/ymax

with tf.name_scope('loss'):
    train_loss = relative_loss_criterion(y_, Y)
tf.summary.scalar('train_loss', train_loss)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(train_loss)

merged = tf.summary.merge_all()


# Data loader
relative_depth_info = dict()
path = 'F:/pyWorkspace/diw2/diw/DIW_Annotations/'
with open(path+'DIW_train_val.csv', 'r') as fin:
    relative_sign = {'<': -1, '=': 0, '>': 1}

    lines = fin.readlines()
    data = list()
    for line in lines:
        if len(line.rstrip('\n')) == 0:
            continue
        data.append(line.rstrip('\n'))

    i = 0
    while i < len(data):
        filename = data[i]

        pair = data[i+1].split(',')
        pair[4] = relative_sign[pair[4]]
        pair = [int(v) for v in pair]

        # scale the coordination to input_width and input_height
        ori_img_width = float(pair[5])
        ori_img_height = float(pair[6])

        yA_scale = float(pair[0]-1)/ori_img_height
        xA_scale = float(pair[1]-1)/ori_img_width
        yB_scale = float(pair[2]-1)/ori_img_height
        xB_scale = float(pair[3]-1)/ori_img_width

        yA = min(input_height-1, max(0, math.floor(yA_scale * input_height )))
        xA = min(input_width -1, max(0, math.floor(xA_scale * input_width  )))
        yB = min(input_height-1, max(0, math.floor(yB_scale * input_height )))
        xB = min(input_width -1, max(0, math.floor(xB_scale * input_width  )))

        # check if A and B scaled to the same point
        if (yA == yB) and (xA == xB):#check this
            if yA_scale > yB_scale:
                yA = min(yA+1, input_height-1)
            else:
                yA = max(yA-1, 0)
            if xA_scale > xB_scale:
                xA = min(xA+1, input_width-1)
            else:
                xA = max(xA-1, 0)

        pairs = list()
        pairs.append([int(yA), int(xA), int(yB), int(xB), int(pair[4])])

        relative_depth_info[int(i/2)] = (filename, pairs)
        i += 2


def load_images(indices):
    path = 'F:/pyWorkspace/diw2/diw/'
    images = list()
    for i in range(len(indices)):
        filename = relative_depth_info[indices[i]][0]
        pic = Image.open(path + filename[1:])
        pic = pic.resize((input_width, input_height))
        pic = np.array(pic)

        # check if it is a gray image
        if len(pic.shape) < 3:
            pic = np.expand_dims(pic, 2)
            pic = np.concatenate((pic,)*3, -1)

        images.append(pic)
    return np.stack(images, axis=0)


def load_targets(indices):
    targets = list()
    for i in range(len(indices)):
        targets.append(relative_depth_info[indices[i]][1])
    return np.asarray(targets)


all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
ckpt_dir = 'checkpoints/relative_loss_diw/'
saver = tf.train.Saver(var_list=[v for v in all_variables if "lf_depth" not in v.name and 'loss' not in v.name and 'train' not in v.name])


# train from scrach
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('log/relative_depth_diw', sess.graph)

    sess.run(tf.global_variables_initializer()) #initialize variables 
    #saver.restore(sess, 'checkpoints/backup/diw/model1544498436.376661.ckpt')
    saver.restore(sess, 'checkpoints/backup/NYU/best_model.ckpt')
    print('Model restored')

    num_epochs = 2
    iters = 0
    for epoch in range(num_epochs):
        total_loss = 0.0

        batch_idx = list(relative_depth_info.keys())
        np.random.shuffle(batch_idx)

        i = 0
        while i + batch_size < len(batch_idx):
            indices = np.asarray(batch_idx[i:i+batch_size])
            i += batch_size
            images = load_images(indices)
            targets = load_targets(indices)
            _, loss = sess.run([train_step, train_loss], feed_dict={X: images, Y: targets})
            total_loss += loss

            if iters%500 == 0:
                print('training iters: %d, loss: %f' % (iters, total_loss/100))
                total_loss = 0.0

                summary = sess.run(merged, feed_dict={X: images, Y: targets})
                train_writer.add_summary(summary, iters)
            if iters%1000 == 0:
                save_path = saver.save(sess, ckpt_dir+'model'+str(time.time())+'.ckpt')
            iters += 1

    train_writer.close()
    print('training finished.')


'''
# test
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #initialize variables 
    #saver.restore(sess, 'checkpoints/relative_loss_diw/model1544221507.681115.ckpt')
    saver.restore(sess, 'checkpoints/relative_loss/model1543629675.0780742.ckpt')
    print('Model restored')

    num_epochs = 1001
    for epoch in range(num_epochs):
        total_loss = 0.0

        batch_idx = list(relative_depth_info.keys())
        np.random.shuffle(batch_idx)

        i = 0
        while i + batch_size < len(batch_idx):
            indices = np.asarray(batch_idx[i:i+batch_size])
            i += batch_size
            images = load_images(indices)
            targets = load_targets(indices)
            result = sess.run(depth, feed_dict={X: images, Y: targets})
        
            depth0 = result[0]

            img = np.round(255.0*depth0)
            img = np.concatenate((img,)*3, -1)
            img = Image.fromarray(np.uint8(img))
            img.show()

            print(indices)

            quit()
'''