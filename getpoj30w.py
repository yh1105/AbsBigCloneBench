f=open("recordtraindataPOJ.txt",'r')
ff=open("recordtraindataPOJ30w.txt",'w')
line="123"
k=0
while True:
    line=f.readline()
    ff.write(line)
    k+=1
    if k==300000:
        break
f.close()
ff.close()