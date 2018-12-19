import tensorflow as tf

# 在使用循环神经网络时,希望读入的训练样本是有序的可使用FIFOQueue
# 先创建一个先入先出的队列,初始化队列插入0.1,0.2,0.3三个数字
q = tf.FIFOQueue(3, tf.float32)
init = q.enqueue_many(([0.1, 0.2, 0.3],))  # 此时数据填充并没有完成，而是做出了一个预备工作，真正的工作要在会话中完成。
# 定义出队,+1,入队操作
x = q.dequeue()
y = x + 1
q_add = q.enqueue(y)

with tf.Session() as sess:
    print(sess.run(init))

    quelen = sess.run(q.size())
    print(quelen)
    for i in range(quelen):
        sess.run(q_add)  # 执行两次操作,队列中的值变为0.3,1.1,1.2
        print(q_add)
    for j in range(quelen):
        print(sess.run(q.dequeue()))  # 输出队列的值