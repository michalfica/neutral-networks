from imp import reload
import c4 
reload(c4)

from c4 import Board
from c4 import AgentRandom, AgentMinMaxMC, AgentMC, AgentNetwork, AgentSimpleNetwork

def game(agent_a, agent_b):
    """zwraca 1  - jeśli gracz, który jako pierwszy wykonywał ruch WYGRAŁ 
       zwraca -1 - jeśli gracz, który jako drugi wykonywał ruch WYGRAŁ """
    
    b = Board()
    agents = [agent_a, agent_b]
    moves = []
    
    who = 0
    
    while not b.end():
        m = agents[who].best_move(b)
        b.apply_move(m) 
                          
        who = 1-who
    
    b.print() 
    print (b.result)
    print ()
    
    return b.result
    
def duel(agent_a, agent_b, N):
    score = {1:0, -1:0, 0:0}
    """score[+1] - wynik agenta A 
       score[-1] - wynik agenta B  (liczba wygranych pojedynków)"""
    
    for i in range(N):
        r1 = game(agent_a, agent_b)
        score[r1] += 1
        r2 = game(agent_b, agent_a)
        score[-r2] += 1
    
    s = sum(score.values())
    
    for k in score:
        score[k] /= s    
    print (f'{agent_a.name}: {score[+1]}, {agent_b.name}: {score[-1]}, Draw: {score[0]}')     
    
    
if __name__ == '__main__':

    A = AgentSimpleNetwork()
    B = AgentRandom()

    # A = AgentRandom()
    #A = AgentMC(50)    
    #B = AgentMC(10)
    # B = AgentMinMaxMC(3,10)
    
    duel(A, B, 1000)    