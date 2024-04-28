from colorama import Fore, Style
from copy import deepcopy
import random
import sys

from network_for_c4 import * 
from DataLoader import * 

import torch 

DX = 7
DY = 6
STRENGTH = 10
LEVEL = 3
GAMMA = 0.999

    
coins = [Fore.BLUE + '⬤', Fore.RED + '⬤']

directions = [ (1,0), (0,1), (1,-1), (1,1) ]

EMPTY = 0

class AgentMC:
    def __init__(self, n_of_rollouts):
        self.n_of_rollouts = n_of_rollouts
        self.name = f'MC({self.n_of_rollouts})'
    
    def best_move(self, b):
        ms = b.moves()
        return b.best_move_rollouts(ms, self.n_of_rollouts)
    
class AgentRandom:
    def __init__(self):
        self.name = 'RND'
        
    def best_move(self, b):
        return b.random_move()
    
class AgentMinMaxMC:
    def __init__(self, level, n_of_rollouts):
        self.level = level
        self.n_of_rollouts = n_of_rollouts
        self.name = f'MM_MC({self.level}, {self.n_of_rollouts})'
    
    def best_move(self, b):
        return b.best_move(self.level, self.n_of_rollouts)
    
def fun_function(x):
    return [1, 2, 3]

class AgentNetwork:
    def __init__(self) -> None:
        self.name = 'NeuralNetwork'
        self.model = find_network()
        self.model.eval()

    def best_move(self, b):
        potential_moves = b.moves()

        best_move, best_value = potential_moves[0], 0 
        for move in potential_moves:

            if b.is_winning(move):
                return move
            
            b.apply_move(move)
            board_after_move = self.encode_board(b.board)
            b.undo_move(move)

            with torch.no_grad():
                x = torch.tensor(board_after_move)
                x = x[None, :, :, :]

                out = self.model(x)
                value_for_this_move = out[0][1].item()

            if value_for_this_move > best_value:
                best_move  = move
                best_value = value_for_this_move
        return best_move       

    def encode_board(self, b):
        """!!! WAŻNE ZAŁOŻENIE !!! - zakładam że jako 1 gracz zaczynam i moje monety sa w polach oznaczonych przez +1 """
        encoded_board = torch.zeros([2, 6, 7])
        for y in range(DY):
            for x in range(DX):
                #  pole planszy 
                who = b[y][x]
                
                if who == 1: 
                    who = 0
                else:
                    who = 1

                encoded_board[who][6-1-y][x] = 1
        return encoded_board

class AgentSimpleNetwork:
    def __init__(self) -> None:
        self.name = 'NeuralNetwork'
        self.model = find_simple_network()
        self.model.eval()
    
    def best_move(self, b):
        potential_moves = b.moves()
        best_move, best_value = potential_moves[0], 0 
        for move in potential_moves:

            if b.is_winning(move):
                return move
            
            b.apply_move(move)
            board_after_move = self.encode_board(b.board, b.hs)
            b.undo_move(move)

            with torch.no_grad():
                x = torch.tensor(board_after_move)
                x = x[None, :]

                out = self.model(x)
                value_for_this_move = out[0][1].item()

            if value_for_this_move > best_value:
                best_move  = move
                best_value = value_for_this_move
        return best_move  

    
    def encode_board(self, b, hs):
        # potrzebuje:
        # plansze z 1 i 2 w polach 
        # wiedzieć kto zaczynał ?? 
        #  kto ma teraz ruch? ja
        # kim jestem graczem pierwszym czy drugim? 
        # zliczyć mogę ile jest jedynek ile -1 tego czego mniej  - takie są moje pionki 
        liczba_jedynke, liczba_minus_jedynek = 0, 0 
        mapped_board = torch.zeros([6, 7])
        for row in range(6):
            for col in range(7):
                # print(f"mb = {mapped_board[5-row][col]}")
                # print(f"b.shape = {len(b), len(b[0])}")
                # print(f"{row, col} - {5-row, col}")
                # print(f"b = {b[row][col]}")
                mapped_board[row][col] = b[5-row][col]
                if b[5-row][col]==-1:
                    mapped_board[row][col] = 2 
                    liczba_minus_jedynek += 1 
                if b[5-row][col]==1:
                    liczba_jedynke += 1 

        if liczba_jedynke > liczba_minus_jedynek:
            ktorym_graczem_jestem = 1
        else:
            ktorym_graczem_jestem = 2 
          
        encoded_board = torch.zeros([8], dtype=torch.float32)
        encoded_board[0] = 0 
        if ktorym_graczem_jestem==1: encoded_board[1] = 2
        if ktorym_graczem_jestem==2: encoded_board[1] = 1 
        helper_ = C4DataSet(10,10)
        horiz, verti = helper_.compute_simple_triples(board=b, cnt=hs, winner=ktorym_graczem_jestem) 
        encoded_board[2] = horiz
        encoded_board[3] = verti
        encoded_board[4] = helper_.compute_catty_corner_triples(board=b, cnt=hs, winner=ktorym_graczem_jestem)
        encoded_board[5] = helper_.compute_verti_holes(b, winner=ktorym_graczem_jestem)
        encoded_board[6] = helper_.compute_catty_corner_holes(b, winner=ktorym_graczem_jestem)
        encoded_board[7] = helper_.compute_pairs(b, winner=ktorym_graczem_jestem)
        return encoded_board
        
    
class Board:
    def __init__(self):
        self.board = [DX * [0] for y in range(DY)]
        self.hs = DX * [0]
        self.who = +1
        self.last_moves = []
        self.move_number = 0
        self.result = '?'
        
    def moves(self):
        return [n for n in range(DX) if self.hs[n] < DY]
        
    def apply_move(self, m):
        h = self.hs[m]
        self.board[h][m] = self.who
        self.hs[m] += 1
        self.who = -self.who
        self.last_moves.append(m)
        self.move_number += 1
        
    def undo_move(self, m):
        h = self.hs[m]
        self.board[h-1][m] = EMPTY
        
        self.hs[m] -= 1
        self.who = -self.who
        self.last_moves.pop()
        self.move_number -=1
                
        
    def print(self):
        for raw in self.board[::-1]:
            for x in range(DX):
                if raw[x] == EMPTY:
                    print ('  ', end='')
                else:
                    r = (raw[x] + 1) // 2
                    print (coins[r] + ' ', end='')
            print ()
        print (Fore.LIGHTYELLOW_EX + 2 * DX*'‒')
        for i in range(DX):
            if self.last_moves and i == self.last_moves[-1]:
                style = Style.BRIGHT
            else:
                style = Style.NORMAL
            print (style + str(i+1), end=' ')
                
        print ()   
        print ()   
        
    def random_move(self):
        ms = self.moves()
        for m in ms:
            if self.is_winning(m):
                return m
        return random.choice(ms)  
        
    def rollout(self, m):
        while True:
            if self.is_winning(m):
                return self.who
            self.apply_move(m)
            ms = self.moves()
            if ms == []:
                return 99
            m = self.random_move()               
            
    
    def move_value(self, m, n_of_rollouts):       
        value = 0
        who_is_playing = self.who 
        for i in range(n_of_rollouts):
            state = (self.who, self.last_moves[:], self.hs[:], deepcopy(self.board))
            
            r = self.rollout(m)
            if r == who_is_playing:
                value += 1
            if r == -who_is_playing:
                value -= 1
            
            self.who, self.last_moves, self.hs, self.board = state
                           
        return value
        
    def best_move_rollouts(self, ms,  n_of_rollouts):
        #return random.choice(ms)
        return max(ms, key=lambda x:self.move_value(x,  n_of_rollouts))
                
    
    def best_moves(self, level):
        #minimax
        ms = self.moves()
        
        vms = []
        for m in ms:
            if self.is_winning(m):
                return [m]
            self.apply_move(m)    
            vms.append( (self.mini_max(level), m))
            self.undo_move(m)
            
        if self.who == 1:
            min_max = max
        else:
            min_max = min
                        
        v_max,m = min_max(vms)
        
        good_moves = [m for (v,m) in vms if v == v_max]
        return good_moves
        
    def best_move(self, level, n_of_rollouts):
        ms = self.best_moves(level)   
        return self.best_move_rollouts(ms, n_of_rollouts)
                
    def mini_max(self, level):
        if level == 0:
            return 0
        ms = self.moves()
        if not ms:
            return 0
        
        vals = []
        for m in ms:
            if self.is_winning(m):
               return self.who * (GAMMA ** self.move_number)
            self.apply_move(m)
            
            vals.append(self.mini_max(level-1))
            self.undo_move(m)
        if self.who == +1:
            return max(vals)
        return min(vals)    
    
    def last_move_was_winning(self):
        return self.was_winning(self.last_moves[-1])
    
    def end(self):
        if not self.last_moves:
            return False
        if self.last_move_was_winning():
            if len(self.last_moves) % 2 == 0:
                self.result = -1
            else:
                self.result = +1
            return True    
        if len(self.last_moves) == DX*DY:
                self.result = 0
                return True 
        return False 
        
    def vertical_winning(self):
        return self.was_vertical_winning(self.last_moves[-1])
        
    def was_winning(self, m):    
        for dx, dy in directions:
            x,y = m, self.hs[m]-1  # after applying move        
            score = 0
            
            while self.board[y][x] == -self.who:
                score += 1
                x += dx
                y += dy
                if not (0<=x<DX and 0<=y<DY):
                    break
            
            x,y = m, self.hs[m]-1      
            dx = -dx
            dy = -dy
            
            while self.board[y][x] == -self.who:
                score += 1
                x += dx
                y += dy
                if not (0<=x<DX and 0<=y<DY):
                    break
            score -= 1
            
            if score >= 4:
                return True

        return False
 
    def was_vertical_winning(self, m):    
        for dx, dy in [(0,1)]:
            x,y = m, self.hs[m]-1  # after applying move        
            score = 0
            
            while self.board[y][x] == -self.who:
                score += 1
                x += dx
                y += dy
                if not (0<=x<DX and 0<=y<DY):
                    break
            
            x,y = m, self.hs[m]-1      
            dx = -dx
            dy = -dy
            
            while self.board[y][x] == -self.who:
                score += 1
                x += dx
                y += dy
                if not (0<=x<DX and 0<=y<DY):
                    break
            score -= 1
            
            if score >= 4:
                return True

        return False
 

            
    def is_winning(self, m):    
        for dx, dy in directions:
            x,y = m, self.hs[m]
            score = 0
            
            while True:
                x += dx
                y += dy
                if not (0<=x<DX and 0<=y<DY):
                    break
            
                if self.board[y][x] == self.who:
                    score += 1
                else:
                    break    
                        
                        
            x,y = m, self.hs[m]
            dx = -dx
            dy = -dy
            
            while True:
                x += dx
                y += dy
                if not (0<=x<DX and 0<=y<DY):
                    break
            
                if self.board[y][x] == self.who:
                    score += 1
                else:
                    break    
            
            score += 1
            
            if score >= 4:
                return True

        return False  
        
                  

    

        
            
        

