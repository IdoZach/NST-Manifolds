import tensorflow as tf

mat1 = [ [1,2,3,5],[4,5,6,5],[7,8,9,5]]
mat2 = [ [1,22,33,5],[44,55,6,5],[7,898,9,5]]
mat3 = [ [1,2,13,5],[14,15,16,5],[17,81,91,5]]
x = tf.Variable(initial_value=[mat1,mat2,mat3,mat3])

print x
x = tf.expand_dims(x,3)
y = tf.extract_image_patches(x,[1,2,2,1],[1,1,1,1],[1,1,1,1],'VALID')
print y
op=tf.initialize_all_variables()
sess=  tf.Session()
sess.run(op)
sess.run(y[0])

