import os 
import json
import time 
import numpy as np 
import scipy.sparse as sp
from .dataloader import DataLoader

class DataLoaderPolyvore(DataLoader):
    def __init__(self, data_path):
        super(DataLoaderPolyvore, self).__init__(data_path)
    
    def init_phase(self, phase):
        print("Init Phase: {}".format(phase))
        assert phase in ('train', 'valid', 'test')
        np.random.seed(1234)

        adjacency_file = os.path.join(self.data_path, 'adj_{}.npz'.format(phase))
        feature_file = os.path.join(self.data_path, 'X_{}.npz'.format(phase))
        question_file = os.path.join(self.data_path, 'questions_test.json')
        question_resample_file = os.path.join(
            self.data_path, 'questions_RESAMPLED_test.json')

        adj_matrix = sp.load_npz(adjacency_file).astype(np.int32)
        node_features = sp.load_npz(feature_file)
        with open(question_file) as f:
            self.questions = json.load(f)
        with open(question_resample_file) as f:
            self.questions_resampled = json.load(f)

        setattr(self, '{}_adj'.format(phase), adj_matrix)
        setattr(self, '{}_features'.format(phase), node_features)

        # get lower tiangle of the adj matrix to avoid duplicate edges
        # since adj is symmetric
        setattr(self, 'lower_{}_adj'.format(phase), sp.tril(adj_matrix).tocsr())

    def get_phase(self, phase):
        """
        Note that data obtained by this method in test phase can be used for model selection purpose.
        It does not correspond to any task which we'd like to perform.
        """
        print("get phase: {}".format(phase))
        assert phase in ('train', 'valid', 'test')

        lower_adj = getattr(self, 'lower_{}_adj'.format(phase))

        # get positive edges 
        pos_row_idx, pos_col_idx = lower_adj.nonzero()
        pos_labels = np.array(lower_adj[pos_row_idx, pos_col_idx]).squeeze()
        # split the positive edges into two parts
        # part 1 is used for message passing - GNN 
        # part 2 is used as prediction target, i.e., we predict connection prob
        # on these edges based on part 1 edges and all node features. Then, loss
        # is evaluated on part 2 edges to update model.
        # This prevents ground truth information leakage to the model !!!
        print("Split positive edges ... ", end='\t')
        num_pos = len(pos_labels)
        perm = np.arange(num_pos)
        np.random.shuffle(perm)
        pos_labels, pos_row_idx, pos_col_idx = pos_labels[perm], pos_row_idx[perm], pos_col_idx[perm]

        num_eval_loss = num_pos // 2
        # positive edges on which we compute loss 
        loss_pos_labels, loss_pos_row_idx, loss_pos_col_idx = pos_labels[:num_eval_loss], pos_row_idx[:num_eval_loss], pos_col_idx[:num_eval_loss]
        # positive edges used for message passing in GNN
        mp_pos_labels, mp_pos_row_idx, mp_pos_col_idx = pos_labels[num_eval_loss:], pos_row_idx[num_eval_loss:], pos_col_idx[num_eval_loss:]
        print("Done!")

        # get negative edges
        print("Sampling negative edges ... ", end='\t')
        # to balance training data, we set the number of sampled negative edges == number of positive edges used for loss evaluation
        num_neg_train = loss_pos_labels.shape[0]
        neg_labels = np.zeros((num_neg_train,))
        neg_row_idx = np.zeros((num_neg_train,))
        neg_col_idx = np.zeros((num_neg_train,))

        s_time = time.time()
        for i in range(num_neg_train):
            r_idx, c_idx = self.get_negative_training_edges(lower_adj.shape[0], lower_adj)
            neg_row_idx[i] = r_idx 
            neg_col_idx[i] = c_idx 
        print("Done!")
        print("Time elapsed: {}\n".format(time.time() - s_time))

        # build a dummy graph for GNN. In its adjacency matrix, only positive_edges for message passing is marked as positive, i.e., connected, 
        # the other edges are simply missing in the dummy graph.
        # note that since the graph is undirected, the dummpy adjacency matrix should be 
        # symmetric.
        adj = sp.csr_matrix(
            (
                np.hstack([mp_pos_labels, mp_pos_labels]),   # values
                (np.hstack([mp_pos_row_idx, mp_pos_col_idx]), np.hstack([mp_pos_col_idx, mp_pos_row_idx]))  # (row idx, col idx)
            ),
            shape=(lower_adj.shape[0], lower_adj.shape[1])
        )
        # remove zeros in adj 
        adj.eliminate_zeros()

        # create ground-truth labels for edges to predict
        eval_labels = np.append(loss_pos_labels, neg_labels)
        eval_row_idx = np.append(loss_pos_row_idx, neg_row_idx)
        eval_col_idx = np.append(loss_pos_col_idx, neg_col_idx)

        # return node features, dummy message passing adjacency matrix, ground-truth labels, row and col idx of edges to evaluate loss
        return getattr(self, '{}_features'.format(phase)), adj, eval_labels, eval_row_idx, eval_col_idx 
    
    def get_test_questions(self, resample=False):
        """
        Return the Fill-in-the-Blank questions in the form of node indecies.
        Each question in self.questions or self.questions_RESAMPLED contains N * 4 edges to predict, where N is the number of items already in the outfit
        self.questions or self.questions_resampled:
        [[q, a, a_pos, desired_pos], ... ]
        q - question: [item1_idx, item2_idx, ... , itemN_idx]
        a - answer: [item1_idx, item2_idx, item3_idx, item4_idx]
        a_pos - pos of answers: [item1_pos, item2_pos, item3_pos, item4_pos]
        desired_pos - ground-truth pos of missing item

        We'd like to transform the FITB problem into a link prediction problem. Basically, we test ever edge between
        each question item and an possible item. For convenience, the above question is flattened into the following
        format:
        [(start_node, end_node), ... ]
        which is a list of edges to predict

        """
        flat_questions = []
        gt_labels = []
        q_ids = []
        
        questions = self.questions if not resample else self.questions_resampled

        # convert questions
        for q_id, question in enumerate(questions):
            for q_item_idx in question[0]:  # outfit item index
                for i, a_item_idx in enumerate(question[1]):  # answer item index
                    flat_questions.append((q_item_idx, a_item_idx))
                    if i == 0: # correct choice
                        gt_labels.append(1)
                    else:
                        gt_labels.append(0)
                    # TODO: add position check here 
                    # set question id 
                    q_ids.append(q_id)
        
        assert len(flat_questions) == len(gt_labels) and len(q_ids) == len(gt_labels)

        flat_questions = np.array(flat_questions)
        gt_labels = np.array(gt_labels)
        q_ids = np.array(q_ids)  # record to which question does each query edge belong to

        # now build message passing adjacency matrix by removing the edges to predict in the FITB task
        # Specifically, since an answer item may appear in an outfit in the test data, it hence connect to other
        # question items in this outfit in the resulting test graph. However, this edge is one of our prediction
        # target. We need remove it from the test graph.
        lower_adj = getattr(self, 'lower_{}_adj'.format('test'))

        full_adj = lower_adj + lower_adj.transpose()
        full_adj = full_adj.tolil()
        for edge, label in zip(flat_questions, gt_labels):
            u, v = edge
            # disconnect this edge
            full_adj[u, v] = 0 
            full_adj[v, u] = 0 
        
        full_adj = full_adj.tocsr()
        full_adj = full_adj.eliminate_zeros()

        # make sure no ground-truth leakage 
        for u, v in flat_questions:
            assert full_adj[u, v] == 0 and full_adj[v, u] == 0 
        
        # message-passing adjacency matrix, source node of edges, end node of edges, edge ground truth labels, question id 
        return full_adj, flat_questions[:, 0], flat_questions[:, 1], gt_labels, q_ids
    
    def yield_test_questions_K_edges(self, resample=False, K=1, subset=False, expand_outfit=False):
        """
        This method yields questions, each of them with their own message passing adjacency matrix. This message passing adjacency matrix considers only 
        edges beteen question item nodes or additional K edges starting from each of the answer and query (optional) nodes. It provies questions similar
        to the practical scenarios in applications and is hence scalable to big graphs. You should always prefer this method for FITB problem over 
        get_test_questions. 

        Each question in self.questions or self.questions_RESAMPLED contains N * 4 edges to predict, where N is the number of items already in the outfit
        self.questions or self.questions_resampled:
        [[q, a, a_pos, desired_pos], ... ]
        q - question: [item1_idx, item2_idx, ... , itemN_idx]
        a - answer: [item1_idx, item2_idx, item3_idx, item4_idx]
        a_pos - pos of answers: [item1_pos, item2_pos, item3_pos, item4_pos]
        desired_pos - ground-truth pos of missing item

        Args:
            K: number of edges to expand for each question node
            resample: if True, use the resampled version
            subset: if True, use only a subset of outfits as the query, and use the rest as links 
                    to the choices.
            expand_outfit: whether to include question items during the BFS expansion process (K>0).
        """
        assert K >= 0

        questions = self.questions if not resample else self.questions_resampled
        # check if we have test_adj available
        assert hasattr(self, '{}_adj'.format('test'))
        n_nodes = self.test_adj.shape[0]

        for question in questions:
            outfit_idx = []
            ans_idx = []
            gt_labels = []

            if subset:
                # sample at most 3 items in the outfit
                outfit_subset = np.random.choice(question[0], size=3, replace=False)
            else:
                outfit_subset = question[0]
            
            for q_item_idx in outfit_subset:
                for idx, a_item_idx in enumerate(question[1]):
                    outfit_idx.append(q_item_idx)
                    ans_idx.append(a_item_idx)
                    gt_labels.append(1 if idx == 0 else 0)
            
            # create an adjacency matrix to denote a graph with only outfit edges
            question_adj = sp.csr_matrix((n_nodes, n_nodes))  # initialize an adjacency matrix with all entries zero.
            question_adj = question_adj.tolil()

            # only set the outfit edges to be 1, i.e., connecting outfit items
            if not expand_outfit:
                for i, u in enumerate(outfit_subset[:-1]):
                    for v in outfit_subset[i+1:]:
                        question_adj[u, v] = 1
                        question_adj[v, u] = 1
            
            if K > 0:
                available_adj = self.test_adj.copy().tolil()
                
                # remove edges connecting question items
                for i, u in enumerate(question[0][:-1]):
                    for v in question[0][i+1:]:
                        available_adj[u, v] = 0
                        available_adj[v, u] = 0
                
                # remove edges connecting question items and answer choices
                for u, v in zip(outfit_idx, ans_idx):
                    available_adj[u, v] = 0
                    available_adj[v, u] = 0 

                # add edges connecting items in the outfit 
                if expand_outfit:
                    for i, u in enumerate(outfit_subset[:-1]):
                        for v in outfit_subset[i+1:]:
                            available_adj[u, v] = 1
                            available_adj[v, u] = 1 
                
                available_adj = available_adj.tocsr()
                available_adj.eliminate_zeros()

                # find nodes to expand. By default, we only expand edges for the answer choices.
                node2expand = ans_idx[:4]  # use 4 since each question has only 4 possible answer choices.

                if expand_outfit:
                    # we also expand edges for outfit question items
                    node2expand.extend(outfit_subset)
                
                for node in node2expand:
                    edges = self.sample_K_edges(available_adj, node, K)

                    for u, v in edges:
                        question_adj[u, v] = 1
                        question_adj[v, u] = 1
            
            question_adj = question_adj.tocsr()

            # query edge - question item <-> answer item
            # question adjacency matrix, start idx of query edge, end idx of query edge, ground truth label of query edge
            yield question_adj, np.array(outfit_idx), np.array(ans_idx), np.array(gt_labels)
            
    def sample_K_edges(self, adj, idx, K):
        """
        Returns a list of K edges, sampled using BFS starting from node idx
        We use BFS since neighbors closer to the target node usually impact more
        on it.
        """
        def bfs(root, adj, visited, K, edges):
            queue = [root]
            while queue:
                node = queue.pop(0)
                if node not in visited:
                    visited.add(node)
                    neighbors = list(adj[node].nonzero()[1])
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            # this edge has never been visited before
                            edges.append((node, neighbor))
                            queue.append(neighbor)
                        
                        # check if we already find enough edges
                        if len(edges) == K:
                            return 
        visited = set()
        edges = []
        bfs(idx, adj, visited, K, edges)
        assert len(edges) <= K
        return edges
    