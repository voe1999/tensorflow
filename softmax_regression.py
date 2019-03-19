from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import scipy.misc

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

save_dir = 'MNIST_data/'

# 将28*28=784维的向量还原为28*28的图像，并且保存为图片
# for i in range(20):
#   image_array = mnist.train.images[0, :]
#   image_array = image_array.reshape(28, 28)
#   filename = save_dir + 'mnist_train_%d.jpg' % i
#   scipy.misc.toimage(image_array, cmin = 0.0, cmax = 1.0).save(filename)

# print(mnist.train.labels[0, :])


# softmax回归在tensorflow中的实现

# x代表待识别的图片，是一个占位符
x = tf.placeholder(tf.float32, [None, 784])

# W是参数，将一个784维的输入转换为10维的输出
W = tf.Variable(tf.zeros([784, 10]))

# b是另一个参数，偏置项bias
b = tf.Variable(tf.zeros([10]))

# y是模型的输出
y = tf.nn.softmax(tf.matmul(x, W) + b)

# y_是实际的标签
y_ = tf.placeholder(tf.float32, [None, 10])

# y和y_应该越相近越好，用交叉熵来衡量两者的损失
# 以下是定义交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

# 定义完之后，下一步是如何优化损失，可以使用梯度下降法优化损失
# 0.01是学习率 learning rate
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 优化前，需要创建一个session，只有在session中才能优化train_step

session = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 进行1000步梯度下降

for _ in range(1000):
    # 从mnist_train中每次取100个数据，训练1000次
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# y和y_的形状都是(N,10), argmax作用是取出数组中最大值的下标，然后用equal方法比较他们是否相等，返回格式类似[True, True, False, False]
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 用cast方法将比较值转化为float32型，类似[1., 1., 0., 0.]
# 然后用reduce_mean计算数组中所有元素的平均值，相当于模型的预测准确率，例如上面的准确率是0.5
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(session.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
