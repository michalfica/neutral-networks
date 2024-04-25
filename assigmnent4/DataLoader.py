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
            # 1 współrzędna = czy zakończył rozgrywkę
            # 2 współrzędna = czyj ruch 

            sample = torch.zeros([2], dtype=torch.float32)

            if i == len(game)-2: 
                sample[0] = 1
            
            sample[1] = i % 2 + 1

            if i > treshold:
                samples.append((sample.detach().clone(), winner)) 
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