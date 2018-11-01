
import tensorflow as tf
import numpy as np
rng = np.random
# 模拟生成100对数据对, 对应的函数为y = x * 0.1 + 0.3
#x_data = np.random.rand(100).astype("float32")
#y_data = x_data * 0.1 + 0.3

x_data = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
y_data = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = x_data.shape[0]
# 指定w和b变量的取值范围（注意我们要利用TensorFlow来得到w和b的值）
#W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
#b = tf.Variable(tf.zeros([1]))
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")
y = W * x_data + b
X = tf.placeholder("float")
Y = tf.placeholder("float")
pred = tf.add(tf.multiply(X, W), b)
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# 最小化均方误差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# 初始化TensorFlow参数
init = tf.initialize_all_variables()

# 运行数据流图（注意在这一步才开始执行计算过程）
sess = tf.Session()
sess.run(init)

# 观察多次迭代计算时，w和b的拟合值
for step in range(5000):
    sess.run(train)
    if step % 1000 == 0:
        print(step, sess.run(W), sess.run(b))
c = sess.run(cost, feed_dict={X: x_data, Y:y_data})
print("cost=",c)
