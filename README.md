# AoA-for-Reading-Comprehension-on-CNN-CBTest-dataset
I have done some thing to apply on CBTest and CNN dataset. All on the basis of the paper "Attention-over-Attention Neural Networks for Reading Comprehension"
针对AoA这篇论文，之前的源代码只支持读取CNN数据，由于CBTest数据集的结构和形式和CNN数据集差异较大，所以重写了reader.py文件。
reader_CNN.py是原来的读数据代码，reader_CBTest.py是我写的读CBTest数据并转化为tf.records代码。
源代码链接为：https://github.com/OlavHN/attention-over-attention <br>
代码执行顺序为：
1，下载数据集:CNN/Daliy Mail:http://cs.nyu.edu/~kcho/DMQA/和Children’s Book Test（CBTest）:http://www.thespermwhale.com/jaseweston/babi/CBTest.tgz
2，运行reader.py文件，将原数据集保存为.tfrecords文件，方便程序的高效读取
3，运行model.py文件，训练模型(vocab size参数可尽量调大一点).
