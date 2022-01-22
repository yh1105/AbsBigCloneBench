import os
import tensorflow as tf
import numpy as np
import network
from sampleJava import getData_nofinetune,_traverse_treewithid
import javalang
from parameters import EPOCHS, LEARN_RATE
from sklearn.metrics import precision_score, recall_score, f1_score
import time
def getWordEmd(word):
    listrechar = np.array([0.0 for i in range(0, len(listchar))])
    tt = 1
    for lchar in word:
        listrechar += np.array(((len(word) - tt + 1) * 1.0 / len(word)) * np.array(dicttChar[lchar]))
        tt += 1
    return listrechar
def train_model(infile, embeddings):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    num_feats = len(getWordEmd('ForStatement'))
    nodes_node1, nodes_node2, res = network.init_net_nofinetune_one_nohidden(num_feats,200)
    labels_node, loss_node = network.loss_layer(res)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(LEARN_RATE)
    train_step = optimizer.minimize(loss_node)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)  # config=tf.ConfigProto(device_count={'GPU':0}))
    sess.run(tf.compat.v1.global_variables_initializer())
    dictt = {}
    listrec = []
    f = open("flistAbsBCB.txt", 'r')
    lbcblist = f.readline().rstrip('\t').split("\t")
    f.close()
    z = 0
    for ll in lbcblist:
        if not os.path.exists(ll):
            print(ll)
            listrec.append(ll)
            continue
        faa = open(ll, 'r', encoding="utf-8")
        fff = faa.read()
        tree = javalang.parse.parse_member_signature(fff)
        sample, size = _traverse_treewithid(tree)
        if size > 5000 or size < 10:
            z += 1
            listrec.append(ll)
            continue
        dictt[ll] = sample
    print("count")
    print(len(listrec))
    print(len(dictt))
    f=open(infile,'r')
    contentListFileTrain=f.read().rstrip("\n").split("\n")
    print(len(contentListFileTrain))
    f.close()
    for epoch in range(1, EPOCHS + 1):
        #break
        k = 0
        for ltemp in contentListFileTrain:
            l = ltemp.split('\t')
            k += 1
            # print(l)
            # print(listrec)
            if (l[0] in listrec) or (l[1] in listrec):
                continue
            batch_labels=[]
            #print(l)
            nodes1, children1, nodes2, children2, la=getData_nofinetune(l,dictt,embeddings)
            batch_labels.append(la)
            _, err, r = sess.run(
                [train_step, loss_node, res],
                feed_dict={
                    nodes_node1: nodes1,
                    nodes_node2: nodes2,
                    labels_node: batch_labels
                }
            )
            maxnodes = max(len(nodes1[0]), len(nodes2[0]))
            if k % 1000 == 0:
                print('Epoch:', epoch,
                      'Step:', k,
                      'Loss:', err,
                      'R:', r,
                      'Max nodes:', maxnodes
                      )
            #if k>200000:
                #break
        #break
        # if epoch%3!=0:
        #     continue
        correct_labels_dev = []
        predictions_dev = []
        for reci in range(0, 15):
            predictions_dev.append([])
        ff = open("recordtestdataAbsBCB.txt", 'r')
        fw = open("./valueLinearAbs200size/valueLinearAbs" + str(epoch) + ".txt", 'w')
        line = "123"
        while line:
            line = ff.readline().rstrip('\n')
            l = line.split('\t')
            if len(l) != 3:
                break
            if (l[0] in listrec) or (l[1] in listrec):
                continue
            batch_labels=[]
            nodes1, children1, nodes2, children2, la = getData_nofinetune(l, dictt, embeddings)
            batch_labels.append(la)
            k += 1
            output = sess.run([res],
                              feed_dict={
                                  nodes_node1: nodes1,
                                  nodes_node2: nodes2,
                              }
                              )
            correct_labels_dev.append(int(batch_labels[0]))
            fw.write(str(output[0]))
            fw.write(" ")
            threaholder = -0.7
            for i in range(0, 15):
                if output[0] >= threaholder:
                    predictions_dev[i].append(1)
                else:
                    predictions_dev[i].append(-1)
                threaholder += 0.1
        fw.close()
        for i in range(0, 15):
            print("testdata\n")
            print("threa:")
            print(i)
            p = precision_score(correct_labels_dev, predictions_dev[i], average='binary')
            r = recall_score(correct_labels_dev, predictions_dev[i], average='binary')
            f1score = f1_score(correct_labels_dev, predictions_dev[i], average='binary')
            print("recall_test:" + str(r))
            print("precision_test:" + str(p))
            print("f1score_test:" + str(f1score))
        ff.close()


def dfsDict(root):
    global listtfinal
    listtfinal.append(str(root['node']))
    global numnodes
    numnodes+=1
    if len(root['children']):
        pass
    else:
        return
    for dictt in root['children']:
        dfsDict(dictt)
if __name__ == '__main__':
    f = open("sentenceAbsBCBwithid.txt", 'r')
    line = "123"
    listword = []
    while line:
        line = f.readline().rstrip("\n")
        listt = line.split(" ")
        listword.extend(listt)
        listword = list(set(listword))
    f.close()
    #print(len(listword))
    dicttChar = {}
    def _onehot(i, total):
        return [1.0 if j == i else 0.0 for j in range(total)]
    listchar = ['7', 'I', 'E', 'D', 'u', 'C', 'Y', 'W', 'y', '|', '9', '^', 'X', 't', 'a', 'o', 'Z', 'b', 'A', 'J', 'R',
                'w', '?', 'g', '3', '$', 'B', 'l', '5', 'z', 'v', 'T', '2', 'd', '<', 'e', 'M', 'c', 'S', 'm', '4', 'K',
                'O', 'f', 'i', '=', 'Q', '+', 'x', 'N', '1', 'r', 'p', 'G', 'k', '*', 'q', 'L', 'P', '.', 'n', 'j', 'V',
                'U', '6', '/', '%', '8', 'F', 's', '!', '-', '&', '>', 'h', 'H', '0', '_']
    for i in range(0, len(listchar)):
        dicttChar[listchar[i]] = _onehot(i, len(listchar))
    dictfinalem = {}
    t = 0
    for l in listword:
        t += 1
        dictfinalem[l] = getWordEmd(l)
    train_model('recordtraindataAbsBCB30w.txt', dictfinalem)