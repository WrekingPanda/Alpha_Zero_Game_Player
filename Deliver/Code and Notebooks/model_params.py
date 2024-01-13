
A4_MODEL_PARAMS = {"size":4, "action_size":4**4, "num_resBlocks":10, "num_hidden":32}
A4_TRAIN_PARAMS = {"n_iterations":20, "self_play_iterations":100, "mcts_iterations":150, "n_epochs":20, "batch_size":128, "n_self_play_parallel":20, "move_cap":4**3}

A5_MODEL_PARAMS = {"size":5, "action_size":5**4, "num_resBlocks":10, "num_hidden":64} 
A5_TRAIN_PARAMS = {"n_iterations":20, "self_play_iterations":100, "mcts_iterations":150, "n_epochs":20, "batch_size":128, "n_self_play_parallel":20, "move_cap":5**3}

A6_MODEL_PARAMS = {"size":6, "action_size":6**4, "num_resBlocks":20, "num_hidden":64} 
A6_TRAIN_PARAMS = {"n_iterations":20, "self_play_iterations":100, "mcts_iterations":200, "n_epochs":20, "batch_size":128, "n_self_play_parallel":20, "move_cap":6**3}

G7_MODEL_PARAMS = {"size":7, "action_size":7**2+1, "num_resBlocks":20, "num_hidden":64} 
G7_TRAIN_PARAMS = {"n_iterations":10, "self_play_iterations":100, "mcts_iterations":200, "n_epochs":20, "batch_size":128, "n_self_play_parallel":20, "move_cap":7**2.5}

G9_MODEL_PARAMS = {"size":9, "action_size":9**2+1, "num_resBlocks":30, "num_hidden":64} 
G9_TRAIN_PARAMS = {"n_iterations":10, "self_play_iterations":100, "mcts_iterations":200, "n_epochs":20, "batch_size":128, "n_self_play_parallel":20, "move_cap":9**2.5}

MODEL_PARAMS = {"A4":A4_MODEL_PARAMS, "A5":A5_MODEL_PARAMS, "A6":A6_MODEL_PARAMS, "G7":G7_MODEL_PARAMS, "G9":G9_MODEL_PARAMS}
TRAIN_PARAMS = {"A4":A4_TRAIN_PARAMS, "A5":A5_TRAIN_PARAMS, "A6":A6_TRAIN_PARAMS, "G7":G7_TRAIN_PARAMS, "G9":G9_TRAIN_PARAMS}