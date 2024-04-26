from tqdm.auto import tqdm
import torch

class InMemDataLoader(object):
    __initialized = False
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        drop_last=False,
    ):
        batches = []
        for i in tqdm(range(len(dataset))):
            batch = [torch.tensor(t) for t in dataset[i]]
            batches.append(batch)
        tensors = [torch.stack(ts) for ts in zip(*batches)]
        dataset = torch.utils.data.TensorDataset(*tensors)
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_sampler option is mutually exclusive "
                    "with batch_size, shuffle, sampler, and "
                    "drop_last"
                )
            self.batch_size = None
            self.drop_last = None

        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with " "shuffle")

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.RandomSampler(dataset)
                else:
                    sampler = torch.utils.data.SequentialSampler(dataset)
            batch_sampler = torch.utils.data.BatchSampler(
                sampler, batch_size, drop_last
            )

        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.__initialized = True

    def __setattr__(self, attr, val):
        if self.__initialized and attr in ("batch_size", "sampler", "drop_last"):
            raise ValueError(
                "{} attribute should not be set after {} is "
                "initialized".format(attr, self.__class__.__name__)
            )

        super(InMemDataLoader, self).__setattr__(attr, val)

    def __iter__(self):
        for batch_indices in self.batch_sampler:
            yield self.dataset[batch_indices]

    def __len__(self):
        return len(self.batch_sampler)

    def to(self, device):
        self.dataset.tensors = tuple(t.to(device) for t in self.dataset.tensors)
        return self

class C4DataSet():

    def __init__(self, number_of_games=1000, moves_observed=10):
        assert number_of_games <= 20000, "max number of games is 20000 !"
        self.moves = moves_observed
        self.number_of_games = number_of_games
        # UWAGA ! OKAZUJE SIĘ ZE:  A - czyli drugi gracz, B - pierwszy gracz gdy narsyowałem przykłady 
        self.winner_options = {
                    "A" : 1, 
                    "B" : 0,
                    "D" : -1  # remis - który olewam 
                }

    def create_data_samples(self, game, k):
        """zostanie zapisanych ostatnich k stanów gry
            if k == 'all' <- wtedy wszystkie ruchy zapamietuje z danej rozgrywki """
        # UWAGA: pomijam (olewam) rozgrywki ,w których był remis

        if k == "all":
            treshold = -1
        else:      
            treshold   = len(game)- 2 - k

        winner = self.winner_options[game[-1]]
        
        if winner == -1:
            return []
        
        samples = []
        board   = torch.zeros([2, 6, 7], dtype=torch.float32)

        cnt     = torch.zeros([7], dtype=torch.int32)
        for i in range(1, len(game)-1):
            turn = (i-1) % 2
            col = int(game[i])
            row = 6 - 1 - cnt[col]
            
            board[turn][row][col] = 1.0 
            cnt[col] += 1

            if i > treshold:
                samples.append((board.detach().clone(), winner)) 
        return samples


    def compute_simple_triples(self, board, cnt, winner):
        horizontal = 0 
        for i in range(6):
            for j in range(4):
                if (board[i][j] == winner+1 and 
                    board[i][j+1]==winner+1 and 
                    board[i][j+2]==winner+1 and 
                    (cnt[j+3] < 6-i or (j>0 and cnt[j-1] < 6-i))):
                    horizontal += 1  
                    print(f"hori {i}, {j}")
        vertical = 0 
        for j in range(7):
            for i in range(1,4):
                if (board[i][j]==winner+1 and 
                    board[i+1][j]==winner +1 and 
                    board[i+2][j]==winner +1 and 
                    cnt[j]==6-i):
                    vertical += 1 
                    print(f"verti {i}, {j}")
        return (horizontal, vertical)
    
    def compute_catty_corner_triples(self, board, cnt, winner):
        triples = 0 
        #  trójki po skosie na prawo
        for i in range(3,6):
            for j in range(0,4):
                if (board[i][j]==winner+1 and 
                    board[i-1][j+1]==winner+1 and 
                    board[i-2][j+2]==winner+1 and 
                    cnt[j+3] <= 8-i):
                    triples += 1 
                    print(f"triple skos prawo {i}, {j}")
        # trójki po skosie na lewo 
        for i in range(3,6):
            for j in range(3,7):
                if (board[i][j]==winner+1 and 
                    board[i-1][j-1]==winner+1 and 
                    board[i-2][j-2]==winner+1 and 
                    cnt[j-3] <= 8 - i):
                    triples += 1 
                    print(f"triple skos lewo {i}, {j}")
        return triples 

    def compute_verti_holes(self, board, winner):
        holes = 0 
        for i in range(6):
            for j in range(4):
                if ((board[i][j]==winner+1 and board[i][j+1]==winner+1 and board[i][j+2]==0 and board[i][j+3]==winner+1) or
                    (board[i][j]==winner+1 and board[i][j+1]==0 and board[i][j+2]==winner+1 and board[i][j+3]==winner+1)):
                    holes += 1 
                    print(f"hole in {i}, {j}")
        return holes 
    
    def create_data_samples_features(self, game, k):
        if k == "all":
            treshold = -1
        else:      
            treshold   = len(game)- 2 - k
            
        winner = self.winner_options[game[-1]]
        if winner == -1:
            return []
        
        samples = []
        board   = torch.zeros([6, 7], dtype=torch.float32)
        cnt     = torch.zeros([7], dtype=torch.int32)
        for i in range(1, len(game)-1):
            turn = (i-1) % 2 + 1 # czyja tura 1 - pierwszego gracza (tego który zaczynał), 2 - drygiegi gracza (wykonywał ruch jako drugi)
            col = int(game[i])
            row = 6 - 1 - cnt[col]

            board[row][col] = turn 
            cnt[col] += 1 

            # próbka - narazie ma rozmiar 2 
            # 1 współ = czy zakończył rozgrywkę
            # 2 współ = czyj ruch 
            # 3 współ = liczba trójek w poziomie, które da się przedłóżyć 
            # 4 współ = liczba trójek w pionie, które da się przedłóżyć 
            # 5 współ = liczba trójek na skos, które da się przedłóżyć
            # 6 współ = liczba dziur między 1 a 2 lub 2 a 1 w pionie 
            
            # TO DO : 
            # 7 współ = liczba dziur między 1 a 2 lub 2 a 1 po skosie 
            # 8 współ = liczba dwójek które da się przedłóżyć 

            sample = torch.zeros([8], dtype=torch.float32)

            if i == len(game)-2: 
                sample[0] = 1
            
            sample[1] = i % 2 + 1
            horiz, verti = self.compute_simple_triples(board, cnt, winner=winner) 
            sample[2] = horiz
            sample[3] = verti
            sample[4] = self.compute_catty_corner_triples(board, cnt, winner=winner)
            sample[5] = self.compute_verti_holes(board, winner=winner)

            if i > treshold:
                samples.append((sample.detach().clone(), winner)) 
        print(f"board =\n {board}")
        return samples
    
    def test_samples(self, game):
        # game = "S331211565510B"
        samples = self.create_data_samples_features(game, "all")
        return samples



    def create_data_set(self, task_nr=1):
        """dataset - lista tupli, pojdeyńczy typel to [0] - tensor o kształcie [2, 6, 7]; [1] - int """
        data_set = []
        with open('games3.txt', 'r') as f:
            for i, line in enumerate(f):
                if i > self.number_of_games-1: 
                    break 
                if task_nr==1:
                    data_set.extend(self.create_data_samples(line.strip(), k=self.moves))
                else:
                    data_set.extend(self.create_data_samples_features(line.strip(), k=self.moves))
        return data_set