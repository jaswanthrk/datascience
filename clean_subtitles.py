text_file = open('subtitles.srt','r')
lines = text_file.readlines()

a = []
x = 2
while x in range(2, len(lines)):
    a.append(lines[x])
    x = x + 4


b = []
for count,x in enumerate(x):
    b.append(x[:-1])

file = open('subs.txt','w+')
c = ' '.join(b)
file.write(c)
file.close()

#d = []
#for cnt, x in enumerate(c):
#    if x == '\\' and c[cnt+1] == "\'":
#        d.append(c[cnt+1])
#    if x == "\'":
#        d.append('')
#    else:
#        d.append(x)

#e = ''.join(d)
