import math
import openpyxl
import numpy as np
from openpyxl import load_workbook
import filter
import DNN

max=10000000000000

#Record Path
def getP(A:[],i):
    W = A[:]
    W.append(i)
    return  W

# Find the nodes that need to be changed
def getN(R:[],a0):
    xa=len(a0)-1
    for i in range(0, len(R[xa])):
      if R[xa][i][0] == a0:
          ya=i

    return  xa,ya

# Get Value
def getV(path,a1,maxaccuracy):
    print(a1)
    p = []
    for i in range(0, len(a1)):
        p.append(10 * a1[i])
    print(p)
    V,maxaccuracy=DNN.net(p, path, maxaccuracy)
    V = V*100
    return  V

# Action
#Take a step forward
def advance_step(S:[],i):
    W=S[:]
    W[i] = W[i] +1.0
    return  W

#Take a step back
def backward_step(S:[],i):
    W = S[:]
    W[i] = W[i] -1.0
    return  W

#Rollout
def rollout(R,a0,V):
    for i in range(0, len(a0)):
        x, y = getN(R, a0[0:i+1])
        R = changeV(R, x, y, V)
        R = changeN(R, x, y)
    return  R

#Change Value
def changeV(R:[],x,y,V):
    W = R[:]
    W[x][y][2]+= V
    return  W

#Increase the number of visits
def changeN(R:[],x,y):
    W = R[:]
    W[x][y][3]+= 1
    return  W

# From leaf nodes to non leaf nodes
def changeF(R:[],x,y):
    W = R[:]
    W[x][y][4] = 1
    return  W

# Print Monte Carlo tree
def Print(R:[]):
    print('Monte Carlo Tree')
    for i in range(0, len(R)):
        print(R[i])
    print()


# Monte Carlo tree
def MCTS(path,c):

        wb = openpyxl.Workbook()
        sheet = wb.active

        # Record
        Record=[[]]
        # print(Record)

        # root node
        Nedo0=[]
        # path
        P=[0]

        # State
        S=[]
        Num=len(c)
        for i in range(0,Num):
            # Not use the filter
            #S.append(0)
            # Use the filter
            S.append(c[i])

        # Value
        V=0
        # number of visit
        N=1
        # Is it a leaf node
        F=0
        Nedo0.append(P)
        Nedo0.append(S)
        Nedo0.append(V)
        Nedo0.append(N)
        Nedo0.append(F)

        Record[0].append(Nedo0)
        a=Record[0][0]

        maxaccuracy=0

        while True:

            # current node
            # print('a:',a)
            if len(Record)>=35:
                # Print(Record[1][0])
                wb.save('./output/recordx.xlsx')
                break

            # Determine whether it is a leaf node
            if a[4]==0:
                # Determine if the number of visit is zero
                if a[3]!=0:

                    # A is no longer a leaf node, starting to expand
                    x, y = getN(Record,a[0])
                    Record=changeF(Record,x,y)

                    if len(Record) <= len(a[0]):
                        Record.append([])

                    for i in range (0,2*Num):
                        Nedoi=[]
                        Pi=getP(a[0],i)

                        if i%2==0:
                            Si = advance_step(a[1], int(i / 2))
                        else:
                            Si = backward_step(a[1], int(i / 2))

                        # Value
                        Vi = 0
                        # number of visit
                        Ni = 0
                        # Is it a leaf node
                        Fi = 0
                        Nedoi.append(Pi)
                        Nedoi.append(Si)
                        Nedoi.append(Vi)
                        Nedoi.append(Ni)
                        Nedoi.append(Fi)

                        # The first node of the expansion is the new node
                        if i==0:
                            b=Nedoi[:]
                            sheet.append(Nedoi[1])
                        Record[len(a[0])].append(Nedoi)



                    a=b[:]
                    # Get the current node value
                    V=getV(path,a[1],maxaccuracy)
                    # Rollout
                    Record=rollout(Record,a[0],V)

                    #restart
                    a=Record[0][0]

                # Not visited before
                else:
                    # Get the current node value
                    V = getV(path,a[1],maxaccuracy)
                    # rollout
                    Record = rollout(Record, a[0], V)
                    # restart
                    a = Record[0][0]

            # Not a leaf node
            else:

                 # Select the child node with the highest UCB value
                 flag = 0
                 USB1Si = -10000000
                 N=Record[0][0][3]
                 c=2

                 for i in range(0, len(Record[len(a[0])])):
                     if a[0]==Record[len(a[0])][i][0][0:len(a[0])]:
                        V = Record[len(a[0])][i][2]
                        n = Record[len(a[0])][i][3]
                        #print(Record[len(a[0])][i],V,c,N,n)
                        if n==0:
                            U=max
                        else:
                            U=V/n+c*math.sqrt(math.log(N)/n)
                        if U>USB1Si:
                            USB1Si=U
                            flag = i

                 #print(flag)
                 a=Record[len(a[0])][flag]

if __name__ == '__main__':

    path = './data/T/E1.xlsx'
    #path = './data/W.xlsx'
    c = filter.getC(path)
    MCTS(path,c)







