import random, math

n = 5
ants = 5
alpha, beta, rho, Q = 1, 5, 0.5, 100
iters = 20

dist = [
    [0, 2, 2, 5, 7],
    [2, 0, 4, 8, 2],
    [2, 4, 0, 1, 3],
    [5, 8, 1, 0, 2],
    [7, 2, 3, 2, 0]
]
pher = [[1]*n for _ in range(n)]

def prob(i,j,v):
    return 0 if j in v else (pher[i][j]**alpha)*((1/dist[i][j])**beta)

def next_city(i,v):
    p=[prob(i,j,v) for j in range(n)]
    s=sum(p)
    if s==0: return random.choice([j for j in range(n) if j not in v])
    r=random.random(); c=0
    for j in range(n):
        c+=p[j]/s
        if r<=c: return j

def tour_len(t):
    return sum(dist[t[i]][t[i+1]] for i in range(n-1))+dist[t[-1]][t[0]]

best,bestL=None,math.inf

for it in range(iters):
    tours=[]
    for _ in range(ants):
        t=[random.randint(0,n-1)]
        while len(t)<n: t.append(next_city(t[-1],t))
        tours.append(t)
    for i in range(n):
        for j in range(n):
            pher[i][j]*=(1-rho)
    for t in tours:
        L=tour_len(t)
        for i in range(n):
            a,b=t[i],t[(i+1)%n]
            pher[a][b]+=Q/L; pher[b][a]+=Q/L
        if L<bestL: best,bestL=t,L
    print(f"Iteration {it+1:2d}: Best Length = {bestL}")

print("\nFinal Best Tour:", best)
print("Final Best Length:", bestL)
