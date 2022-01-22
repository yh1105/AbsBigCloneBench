import os
import tensorflow as tf
import numpy as np
import network
from sampleJava import _traverse_tree_noid
from sampleJava import getData_finetune
import javalang
from parameters import EPOCHS, LEARN_RATE
from sklearn.metrics import precision_score, recall_score, f1_score

def train_model(infile, embeddings):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    num_feats = 100
    nodes_node1, nodes_node2, res = network.init_net_finetune_oneconv(num_feats,embeddingg,100)
    labels_node, loss_node = network.loss_layer(res)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(LEARN_RATE)
    train_step = optimizer.minimize(loss_node)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)  # config=tf.ConfigProto(device_count={'GPU':0}))
    sess.run(tf.compat.v1.global_variables_initializer())
    dictt = {}
    listrec = []
    f = open("flistBCB.txt", 'r')
    line = f.readline().rstrip('\t')
    l = line.split('\t')
    z = 0
    for ll in l:
        if not os.path.exists(ll):
            listrec.append(ll)
            continue
        faa = open(ll, 'r', encoding="utf-8")
        fff = faa.read()
        tree = javalang.parse.parse_member_signature(fff)
        sample, size = _traverse_tree_noid(tree)
        if size > 3000 or size < 10:
            z += 1
            listrec.append(ll)
            continue
        dictt[ll] = sample

    f.close()
    #print(dictt)
    print("wuxiaogeshu:", z)
    for epoch in range(1, EPOCHS + 1):
        f = open(infile, 'r')
        line = "123"
        k = 0
        aaa = 1
        while line:

            line = f.readline().rstrip('\n')
            l = line.split('\t')
            #print(l)
            if len(l) != 3:
                break
            k += 1
            if (l[0] in listrec) or (l[1] in listrec):
                continue
            nodes11,children1,nodes22,children2,batch_labels=getData_finetune(l,dictt,embeddings)
            _, err, r = sess.run(
                [train_step, loss_node, res],
                feed_dict={
                    nodes_node1: nodes11,
                    nodes_node2: nodes22,
                    labels_node: batch_labels,
                }
            )
            if aaa % 1000 == 0:
                print('Epoch:', epoch,
                      'Step:', aaa,
                      'Loss:', err,
                      'R:', r,
                      )
            # print('Epoch:', epoch,
            #       'Step:', aaa,
            #       'Loss:', err,
            #       'R:', r,
            #       )
            aaa += 1
        f.close()
        correct_labels_dev = []
        predictions_dev = []
        for i in range(0, 18):
            predictions_dev.append([])
        ff = open("recordtestdataBCB.txt", 'r')
        line = "123"
        k = 0
        while line:
            line = ff.readline().rstrip('\n')
            l = line.split('\t')
            if len(l) != 3:
                break
            k += 1
            label = l[2]
            if (l[0] in listrec) or (l[1] in listrec):
                continue
            nodes11,children1,nodes22,children2,_=getData_finetune(l,dictt,embeddings)
            output = sess.run([res],
                              feed_dict={
                                  nodes_node1: nodes11,
                                  nodes_node2: nodes22,
                              }
                              )
            correct_labels_dev.append(int(label))
            threaholder = -0.9
            for i in range(0, 18):
                if output[0] >= threaholder:
                    predictions_dev[i].append(1)
                else:
                    predictions_dev[i].append(-1)
                threaholder += 0.1
        for i in range(0, 18):
            print("threholderr:")
            print(-0.9+i*1.0)
            p = precision_score(correct_labels_dev, predictions_dev[i], average='binary')
            r = recall_score(correct_labels_dev, predictions_dev[i], average='binary')
            f1score = f1_score(correct_labels_dev, predictions_dev[i], average='binary')
            print("recall_test:" + str(r))
            print("precision_test:" + str(p))
            print("f1score_test:" + str(f1score))
        ff.close()


if __name__ == '__main__':
    dictt = {}
    dictta = {}
    listta = list()
    def _onehot(i, total):
        return [1.0 if j == i else 0.0 for j in range(total)]
    feature_size = 100
    listchar = ['Statement', 'SuperMemberReference', 'FormalParameter', 'ElementValuePair', 'SuperMethodInvocation',
                'ArrayCreator', 'ArraySelector', 'Literal', 'ConstructorDeclaration', 'SuperConstructorInvocation',
                'VoidClassReference', 'InnerClassCreator', 'VariableDeclaration', 'ForStatement', 'TernaryExpression',
                'ContinueStatement', 'Assignment', 'BasicType', 'SwitchStatement', 'CatchClause', 'BreakStatement',
                'AssertStatement', 'BlockStatement', 'MethodDeclaration', 'Annotation', 'ArrayInitializer',
                'MemberReference', 'EnhancedForControl', 'DoStatement', 'TypeParameter', 'This', 'StatementExpression',
                'SynchronizedStatement', 'TryStatement', 'ClassCreator', 'MethodInvocation',
                'ExplicitConstructorInvocation', 'ReferenceType', 'LocalVariableDeclaration', 'Cast', 'ThrowStatement',
                'ForControl', 'WhileStatement', 'BinaryOperation', 'ElementArrayValue', 'TypeArgument',
                'CatchClauseParameter', 'ClassDeclaration', 'ClassReference', 'FieldDeclaration', 'ReturnStatement',
                'VariableDeclarator', 'SwitchStatementCase', 'IfStatement']
    for l in listchar:
        listta.append(np.random.normal(0, 0.1, 100).astype(np.float32))
    embeddingg = np.asarray(listta)
    embeddingg = tf.Variable(embeddingg)
    for i in range(len(listchar)):
        dictta[listchar[i]] = i
    train_model('recordtraindataBCB30w.txt', dictta)