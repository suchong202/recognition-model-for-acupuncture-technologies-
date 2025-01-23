#%%
from keras.models import load_model
import openpyxl
from openpyxl import load_workbook
import numpy as np
from sklearn.linear_model import Lasso
import Feature
import Read
import os


# RFTR and RDTR or RFLT and RDLT
model_path1 = './model/T/model1.h5'
model1 = load_model(model_path1)
# RFTR or RDTR
model_path2 = './model/T/model2.h5'
model2 = load_model(model_path2)
# RFLT or RDLT
model_path3 = './model/T/model3.h5'
model3 = load_model(model_path3)

# RFTR and RDTR or RFLT and RDLT
model_path4 = './model/V/model1.h5'
model4 = load_model(model_path4)
# RFTR or RDTR
model_path5 = './model/V/model2.h5'
model5 = load_model(model_path5)
# RFLT or RDLT
model_path6 = './model/V/model3.h5'
model6 = load_model(model_path6)


def Tmodel(path,model):
    D = Read.readexecl(path)
    channls = 8
    L = 50
    n = 0
    score = 0
    for j in range(0, len(D[0]) - L):

        P1 = D[0][j:j + L]
        P2 = D[1][j:j + L]

        if Feature.Chu(P1, P2) == 1:
            A = []
            for i in range(0, channls):
                D1 = np.array(D[i][j:j + L])
                A.append(Feature.E(D1))
                A.append(Feature.RMS(D1))
                A.append(Feature.W2(D1))
                A.append(Feature.W3(D1))
                A.append(Feature.W7(D1))
                a, b, c = Feature.findpeaks(D1)
                A.append((a - c) - (c - b))

            input = []
            input.append(A)
            pre = model.predict(input)
            n = n + 1
            score = score + pre[0][0]

    if n == 0:
        return 0.5
    else:
        return score / n


def Vmodel(path, model):
    D = Read.readexecl(path)
    channls = 24
    L = 50

    n = 0
    score = 0
    for j in range(0, len(D[0]) - L):

        V4X = D[9][j:j + L]
        V4Y = D[10][j:j + L]
        V4Z = D[11][j:j + L]
        V8X = D[21][j:j + L]
        V8Y = D[22][j:j + L]
        V8Z = D[23][j:j + L]

        if Feature.Shi(V4X, V4Y, V4Z, V8X, V8Y, V8Z) == 1:

            A = []
            for i in range(0, channls):
                D1 = np.array(D[i][j:j + L])
                A.append(Feature.E(D1))
                A.append(Feature.RMS(D1))
                A.append(Feature.W2(D1))
                A.append(Feature.W3(D1))
                A.append(Feature.W7(D1))
                a, b, c = Feature.findpeaks(D1)
                A.append((a - c) - (c - b))

            input = []
            input.append(A)

            pre = model.predict(input)
            #print(pre[0][0])
            n = n + 1
            score = score + pre[0][0]

    if n == 0:
        return 0.5
    else:
        return score / n


def tactileprediction(path,p,q):

    flag1=Tmodel(path, model1)

    k = 0.5
    Ac1 = 0
    Ac2 = 0

    type=''
    # RFTR and RDTR or RFLT and RDLT
    if flag1 < k:
        p = p + 1
        Ac1 = 1 - abs(flag1 - 0) / abs(k - 0)
        type = type + 'TR'

        flag2 = Tmodel(path, model2)

        if flag2 <= k:
            Ac2 = 1 - abs(flag2 - 0) / abs(k - 0)
            type = 'RF' + type
        else:
            Ac2 = 1 - abs(flag2 - 1) / abs(k - 1)
            type = 'RD' + type

    if flag1 > k:
        p = p + 1
        Ac1 = 1 - abs(flag1 - 1) / abs(k - 1)
        type = type + 'LT'

        flag2 = Tmodel(path, model3)

        if flag2 <= k:
            Ac2 = 1 - abs(flag2 - 0) / abs(k - 0)
            type = 'RF' + type
        else:
            Ac2 = 1 - abs(flag2 - 1) / abs(k - 1)
            type = 'RD' + type

    if type != '':
        h = int(path[-6])
        if type == 'RFTR':
            e = 1
        if type == 'RDTR':
            e = 2
        if type == 'RFLT':
            e = 3
        if type == 'RDLT':
            e = 4

        print(h, e)

        if h == e:
            q = q + 1

        print(q, '/', p, ':Result:' + type, 'layer1-accuracy:' + str(Ac1), 'layer2-accuracy:' + str(Ac2))

    return p, q


def visionprediction(path,p,q):

    flag1 = Vmodel(path, model4)
    print(flag1)

    k = 0.5
    Ac1 = 0
    Ac2 = 0

    type = ''
    # RFTR and RDTR or RFLT and RDLT
    if flag1 < k:
        Ac1 = 1 - abs(flag1 - 0) / abs(k - 0)
        type = type + 'TR'

        flag2 = Vmodel(path, model5)
        print(flag2)

        if flag2 <= k:
            Ac2 = 1 - abs(flag2 - 0) / abs(k - 0)
            type = 'RF' + type
        else:
            Ac2 = 1 - abs(flag2 - 1) / abs(k - 1)
            type = 'RD' + type

    if flag1 > k:
        Ac1 = 1 - abs(flag1 - 1) / abs(k - 1)
        type = type + 'LT'

        flag2 = Vmodel(path, model6)
        print(flag2)

        if flag2 <= k:
            Ac2 = 1 - abs(flag2 - 0) / abs(k - 0)
            type = 'RF' + type
        else:
            Ac2 = 1 - abs(flag2 - 1) / abs(k - 1)
            type = 'RD' + type

    if type != '':
        p=p+1
        h = int(path[-6])
        if type == 'RFTR':
            e = 1
        if type == 'RDTR':
            e = 2
        if type == 'RFLT':
            e = 3
        if type == 'RDLT':
            e = 4

        print(h, e)

        if h == e:
            q = q + 1

        print(q, '/', p, ':Result:' + type, 'layer1-accuracy:' + str(Ac1), 'layer2-accuracy:' + str(Ac2))

    return p, q

def tactileandvisionprediction(path1,path2,p,q):

    flag1 = 0.5*Tmodel(path1, model1)+0.5*Vmodel(path2, model4)

    print(flag1)

    k = 0.5
    Ac1 = 0
    Ac2 = 0

    type = ''
    # RFTR and RDTR or RFLT and RDLT
    if flag1 < k:
        Ac1 = 1 - abs(flag1 - 0) / abs(k - 0)
        type = type + 'TR'

        flag2 = 0.5*Tmodel(path1, model2)+0.5*Vmodel(path2, model3)
        print(flag2)

        if flag2 <= k:
            Ac2 = 1 - abs(flag2 - 0) / abs(k - 0)
            type = 'RF' + type
        else:
            Ac2 = 1 - abs(flag2 - 1) / abs(k - 1)
            type = 'RD' + type

    if flag1 > k:
        Ac1 = 1 - abs(flag1 - 1) / abs(k - 1)
        type = type + 'LT'

        flag2 = 0.5*Tmodel(path1, model3)+0.5*Vmodel(path2, model6)
        print(flag2)

        if flag2 <= k:
            Ac2 = 1 - abs(flag2 - 0) / abs(k - 0)
            type = 'RF' + type
        else:
            Ac2 = 1 - abs(flag2 - 1) / abs(k - 1)
            type = 'RD' + type

    if type != '':
        p = p + 1
        h = int(path1[-6])
        if type == 'RFTR':
            e = 1
        if type == 'RDTR':
            e = 2
        if type == 'RFLT':
            e = 3
        if type == 'RDLT':
            e = 4

        print(h, e)

        if h == e:
            q = q + 1

        print(q, '/', p, ':Result:' + type, 'layer1-accuracy:' + str(Ac1), 'layer2-accuracy:' + str(Ac2))

    return p, q

if __name__ == '__main__':

    a=0
    b=0

    path1 = './data/T/Expert/01/4.xlsx'
    path2 = './data/V/Expert/01/4.xlsx'
    #tactileprediction(path1,a,b)
    visionprediction(path2,a,b)
    tactileandvisionprediction(path1, path2, a, b)
