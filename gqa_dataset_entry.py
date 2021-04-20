"""
This file provide data loading for phase 1 development.
We could ground truth scene graph from this file.

Need to refactor the graph loading components out.
"""
import os
import sys
import pickle
import pathlib
import json
import torch
import Constants
import numpy as np
import torch_geometric
import torchtext


# ROOT_DIR = pathlib.Path('/home/weixin/neuralPoolTest/')
# SCENEGRAPHS = pathlib.Path('/data/weixin/GQA/sceneGraphs/')  # SCENEGRAPHS = ROOT_DIR / 'Downloads' / 'sceneGraphs'
# EXPLAINABLE_GQA_DIR = pathlib.Path('/home/weixin/neuralPoolTest/GraphVQA')


# ROOT_DIR = pathlib.Path('/Users/yanhaojiang/Desktop/cs224w_final/')
# SCENEGRAPHS = pathlib.Path('/Users/yanhaojiang/Desktop/cs224w_final/GraphVQA/sceneGraphs')  # SCENEGRAPHS = ROOT_DIR / 'Downloads' / 'sceneGraphs'
# EXPLAINABLE_GQA_DIR = pathlib.Path('/Users/yanhaojiang/Desktop/cs224w_final/GraphVQA')


ROOT_DIR = Constants.ROOT_DIR
SCENEGRAPHS = ROOT_DIR.joinpath('GraphVQA', 'sceneGraphs')  # SCENEGRAPHS = ROOT_DIR / 'Downloads' / 'sceneGraphs'
EXPLAINABLE_GQA_DIR = ROOT_DIR.joinpath('GraphVQA')

SPLIT_TO_H5_PATH_TABLE = {
    'train_unbiased': '/home/ubuntu/GQA/objectDetection/extract/save_train_feature.h5',
    'val_unbiased': '/home/ubuntu/GQA/objectDetection/extract/save_test_feature.h5',
    'testdev': '/home/ubuntu/GQA/objectDetection/extract/testdev_feature.h5',
    'debug': '/home/ubuntu/GQA/objectDetection/extract/save_train_feature.h5',
}
SPLIT_TO_MODE_TABLE = {
    'train_unbiased': 'train',
    'val_unbiased': 'test',
    'testdev': 'testdev',
    'debug': 'train',
}
SPLIT_TO_PROGRAMMED_QUESTION_PATH_TABLE = {
     'train_unbiased': str(ROOT_DIR / 'GraphVQA/questions/train_balanced_programs.json'),
     'val_unbiased': str(ROOT_DIR / 'GraphVQA/questions/val_balanced_programs.json'),
     'testdev': str(ROOT_DIR / 'GraphVQA/questions/testdev_balanced_programs.json'),
     'debug': str(ROOT_DIR / 'GraphVQA/debug_programs.json'),
     'val_all': str(ROOT_DIR / 'explainableGQA/questions/val_all_programs.json')
}

class GQA_gt_sg_feature_lookup:

    # GT Scene Graph Vocab: to preprocess and numerlicalize text
    SG_ENCODING_TEXT = torchtext.data.Field(sequential=True, tokenize="spacy",
                                            init_token="<start>", eos_token="<end>",
                                            include_lengths=False,
                                            tokenizer_language='en_core_web_sm',
                                            # include_lengths=True,
                                            batch_first=False)

    def __init__(self, split):
        self.split = split
        assert split in SPLIT_TO_H5_PATH_TABLE

        ##################################
        # build vocab for GQA Scene Graph Encoding
        ##################################
        self.build_scene_graph_encoding_vocab()

        ##################################
        # load data based on split
        ##################################
        if split in ['debug', 'train_unbiased']:

            if split == 'debug':
                with open(ROOT_DIR / 'GraphVQA' / 'debug_sceneGraphs.json') as f:
                    self.sg_json_data = json.load(f)

            elif split == 'train_unbiased':
                with open(SCENEGRAPHS / 'train_sceneGraphs.json') as f:
                    self.sg_json_data = json.load(f)
        else:
            assert split in ['val_unbiased', 'testdev', 'val_all']

            if split == 'val_unbiased' or split == 'val_all':
                with open(SCENEGRAPHS / 'val_sceneGraphs.json') as f:
                    self.sg_json_data = json.load(f)

            elif split == 'testdev':
                ##################################
                # We do not have ground truth scene graph for testdev
                ##################################
                self.sg_json_data = None


    def query_and_translate(self, queryID: str, new_execution_buffer: list):
        ##################################
        # handle scene graph part
        ##################################
        sg_this = self.sg_json_data[queryID]
        sg_datum = self.convert_one_gqa_scene_graph(sg_this)

        ##################################
        # Do translation for target object IDs in th execution buffer
        # No need to translate. already based on sorted sg obj_id.
        # TODO: empty object mapping - fix that.
        ##################################

        execution_bitmap = torch.zeros(
            (sg_datum.num_nodes, GQATorchDataset.MAX_EXECUTION_STEP),
            dtype=torch.float32
        )
        instr_annotated_len = min(
            len(new_execution_buffer), GQATorchDataset.MAX_EXECUTION_STEP
        )
        padding_len = GQATorchDataset.MAX_EXECUTION_STEP - instr_annotated_len

        ##################################
        # Build Bitmap based on instructions
        ##################################
        for instr_idx in range(instr_annotated_len):
            execution_target_list = new_execution_buffer[instr_idx]
            for trans_obj_id in execution_target_list:
                execution_bitmap[trans_obj_id, instr_idx] = 1.0

        ##################################
        # Padding Bitmap by copying the last annotation
        ##################################
        for instr_idx in range(instr_annotated_len, instr_annotated_len + padding_len):
            execution_bitmap[:, instr_idx] = execution_bitmap[:, instr_annotated_len - 1]

        sg_datum.y = execution_bitmap

        return sg_datum

    def build_scene_graph_encoding_vocab(self):
        """
        Use torchtext vocab and glove embeddings to encode symbolic tokens in scene graph
        """
        ##################################
        # Build the vocab with tokenized dataset
        # text = SG_ENCODING_TEXT.preprocess(text)
        # # must do preprocess (tokenization) before doing the buld vocab
        ##################################
        def load_str_list(fname):
            with open(fname) as f:
                lines = f.read().splitlines()
            return lines

        tmp_text_list = []
        tmp_text_list += load_str_list(ROOT_DIR / 'GraphVQA/meta_info/name_gqa.txt')
        tmp_text_list += load_str_list(ROOT_DIR / 'GraphVQA/meta_info/attr_gqa.txt')
        tmp_text_list += load_str_list(ROOT_DIR / 'GraphVQA/meta_info/rel_gqa.txt')

        import Constants
        tmp_text_list += Constants.OBJECTS_INV + Constants.RELATIONS_INV + Constants.ATTRIBUTES_INV
        tmp_text_list.append("<self>") # add special token for self-connection
        tmp_text_list = [ tmp_text_list ]

        GQA_gt_sg_feature_lookup.SG_ENCODING_TEXT.build_vocab(tmp_text_list, vectors="glove.6B.300d")
        return GQA_gt_sg_feature_lookup.SG_ENCODING_TEXT

        """
        Scene Graph Encoding:

        using package: https://pytorch-geometric.readthedocs.io/

        input: a scene graph in GQA format,
        output: a graph representation in pytorch geometric format - an instance of torch_geometric.data.Data


        Data Handling of Graphs

        A graph is used to model pairwise relations (edges) between objects (nodes).
        A single graph in PyTorch Geometric is described by an instance of torch_geometric.data.Data,
        which holds the following attributes by default:

        - data.x: Node feature matrix with shape [num_nodes, num_node_features]
        - data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
        - data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        - [Not Applicable Here] data.y: Target to train against (may have arbitrary shape), e.g.,
                                        node-level targets of shape [num_nodes, *] or graph-level
                                        targets of shape [1, *]
        - [Not Applicable Here] data.pos: Node position matrix with shape [num_nodes, num_dimensions]

        """

    def convert_one_gqa_scene_graph(self, sg_this):

        ##################################
        # Make sure that it is not an empty graph
        ##################################
        # assert len(sg_this['objects'].keys()) != 0, sg_this
        if len(sg_this['objects'].keys()) == 0:
            # only in val
            # print("Got Empty Scene Graph", sg_this) # only one empty scene graph during val
            # use a dummy scene graph instead
            sg_this = {
                'objects': {
                    '0': {
                        'name': '<UNK>',
                        'relations': [
                            {
                                'object': '1',
                                'name': '<UNK>',
                            }
                        ],
                        'attributes': ['<UNK>'],
                    },
                    '1': {
                        'name': '<UNK>',
                        'relations': [
                            {
                                'object': '0',
                                'name': '<UNK>',
                            }
                        ],
                        'attributes': ['<UNK>'],
                    },

                }
            }

        SG_ENCODING_TEXT = GQA_gt_sg_feature_lookup.SG_ENCODING_TEXT

        ##################################
        # graph node: objects
        ##################################
        objIDs = sorted(sg_this['objects'].keys()) # str
        map_objID_to_node_idx = {objID: node_idx for node_idx, objID in enumerate(objIDs)}

        ##################################
        # Initialize Three key components for graph representation
        ##################################
        node_feature_list = []
        edge_feature_list = []
        # [[from, to], ...]
        edge_topology_list = []
        added_sym_edge_list = [] # yanhao: record the index of added edges in the edge_feature_list

        ##################################
        # Duplicate edges, making sure that the topology is symmetric
        ##################################
        from_to_connections_set = set()
        for node_idx in range(len(objIDs)):
            objId = objIDs[node_idx]
            obj = sg_this['objects'][objId]
            for rel in obj['relations']:
                # [from self as source, to outgoing]
                from_to_connections_set.add((node_idx, map_objID_to_node_idx[rel["object"]]))
        # print("from_to_connections_set", from_to_connections_set)

        for node_idx in range(len(objIDs)):
            ##################################
            # Traverse Scene Graph's objects based on node idx order
            ##################################
            objId = objIDs[node_idx]
            obj = sg_this['objects'][objId]

            ##################################
            # Encode Node Feature: object category, attributes
            # Note: not encoding spatial information
            # - obj['x'], obj['y'], obj['w'], obj['h']
            ##################################
            # MAX_OBJ_TOKEN_LEN = 4 # 1 name + 3 attributes
            MAX_OBJ_TOKEN_LEN = 12

            # 4 X '<pad>'
            object_token_arr = np.ones(MAX_OBJ_TOKEN_LEN, dtype=np.int_) * \
                SG_ENCODING_TEXT.vocab.stoi[SG_ENCODING_TEXT.pad_token]

            # should have no error
            object_token_arr[0] = SG_ENCODING_TEXT.vocab.stoi[obj['name']]
            # assert object_token_arr[0] !=0 , obj
            if object_token_arr[0] == 0:
                # print("Out Of Vocabulary Object:", obj['name'])
                pass
            # now use this constraint: 1–3 attributes
            # deduplicate

            ##################################
            # Comment out this to see the importance of attributes
            ##################################

            for attr_idx, attr in enumerate(set(obj['attributes'])):
                object_token_arr[attr_idx + 1] = SG_ENCODING_TEXT.vocab.stoi[attr]

            node_feature_list.append(object_token_arr)

            ##################################
            # Need to Add a self-looping edge
            ##################################
            edge_topology_list.append([node_idx, node_idx]) # [from self, to self]
            edge_token_arr = np.array([SG_ENCODING_TEXT.vocab.stoi['<self>']], dtype=np.int_)
            edge_feature_list.append(edge_token_arr)

            ##################################
            # Encode Edge
            # - Edge Feature: edge label (name)
            # - Edge Topology: adjacency matrix
            # GQA relations [dict]  A list of all outgoing relations (edges) from the object (source).
            ##################################


            ##################################
            # Comment out the whole for loop to see the importance of attributes
            ##################################

            for rel in obj['relations']:
                # [from self as source, to outgoing]
                edge_topology_list.append([node_idx, map_objID_to_node_idx[rel["object"]]])
                # name of the relationship
                edge_token_arr = np.array([SG_ENCODING_TEXT.vocab.stoi[rel["name"]]], dtype=np.int_)
                edge_feature_list.append(edge_token_arr)

                ##################################
                # Symmetric
                # - If there is no symmetric edge, add one.
                # - Should add mechanism to check duplicates
                ##################################
                if (map_objID_to_node_idx[rel["object"]], node_idx) not in from_to_connections_set:
                    # print("catch!", (map_objID_to_node_idx[rel["object"]], node_idx), rel["name"])

                    # reverse of [from self as source, to outgoing]
                    edge_topology_list.append([map_objID_to_node_idx[rel["object"]], node_idx])
                    # re-using name of the relationship
                    edge_feature_list.append(edge_token_arr) 

                    # yanhao: record the added edge's index in feature and idx array:
                    added_sym_edge_list.append(len(edge_feature_list)-1)

        ##################################
        # Convert to standard pytorch geometric format
        # - node_feature_list
        # - edge_feature_list
        # - edge_topology_list
        ##################################

        # print("sg_this", sg_this)
        # print("objIDs", objIDs)
        # print("node_feature_list", node_feature_list)
        # print("node_feature_list", node_feature_list)
        # print("node_feature_list", node_feature_list)

        node_feature_list_arr = np.stack(node_feature_list, axis=0)
        # print("node_feature_list_arr", node_feature_list_arr.shape)

        edge_feature_list_arr = np.stack(edge_feature_list, axis=0)
        # print("edge_feature_list_arr", edge_feature_list_arr.shape)

        edge_topology_list_arr = np.stack(edge_topology_list, axis=0)
        # print("edge_topology_list_arr", edge_topology_list_arr.shape)
        del edge_topology_list_arr

        # edge_index = torch.tensor([[0, 1],
        #                         [1, 0],
        #                         [1, 2],
        #                         [2, 1]], dtype=torch.long)
        edge_index = torch.tensor(edge_topology_list, dtype=torch.long)
        # x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        x = torch.from_numpy(node_feature_list_arr).long()
        edge_attr = torch.from_numpy(edge_feature_list_arr).long()
        datum = torch_geometric.data.Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)

        # yanhao: add an additional variable to datum:
        added_sym_edge = torch.LongTensor(added_sym_edge_list)
        datum.added_sym_edge = added_sym_edge


        return datum







class GQATorchDataset(torch.utils.data.Dataset):

    ##################################
    # common config for QA part
    ##################################
    # MAX_EXECUTION_STEP = 9 # strict 9
    # strict 9
    MAX_EXECUTION_STEP = 5

    # QA vocab: to preprocess and numerlicalize text
    TEXT = torchtext.data.Field(sequential=True,
                                tokenize="spacy",
                                init_token="<start>",
                                eos_token="<end>",
                                tokenizer_language='en_core_web_sm',
                                # include_lengths=True,
                                include_lengths=False,
                                # Whether to produce tensors with the batch dimension first.
                                batch_first=False)

    ##################################
    # Init class-level var
    # Prepare short answer vocab
    # read only for using self.XXX to access
    ##################################
    # with open(ROOT_DIR / 'GraphVQA/meta_info/GQA_plural_answer.json') as infile:
    #     answer_plural_inv_map = json.load(infile)
    with open(ROOT_DIR / 'GraphVQA/meta_info/trainval_ans2label.json') as infile:
        ans2label = json.load(infile)
    with open(ROOT_DIR / 'GraphVQA/meta_info/trainval_label2ans.json') as infile:
        label2ans = json.load(infile)
    assert len(ans2label) == len(label2ans)
    for ans, label in ans2label.items():
        assert label2ans[label] == ans

    def __init__(
            self,
            split,
            build_vocab_flag=False,
            load_vocab_flag=False
    ):

        self.split = split
        assert split in SPLIT_TO_H5_PATH_TABLE

        ##################################
        # Prepare scene graph loader (Ground Truth loader or DETR feature loader)
        ##################################

        ### Using Ground Truth Scene Graph
        self.sg_feature_lookup = GQA_gt_sg_feature_lookup(self.split) # Using Ground Truth




        ##################################
        # common config for QA part
        ##################################
        # self.max_src = 30
        # self.max_trg = 80
        # self.num_regions = 32

        ##################################
        # load data based on split
        ##################################
        ##################################
        # load data based on split
        ##################################
        programmed_question_path = SPLIT_TO_PROGRAMMED_QUESTION_PATH_TABLE[split]
        with open(programmed_question_path) as infile:
            self.data = json.load(infile)

        if split in ['debug', 'train_unbiased']:

            ##################################
            # build qa vocab
            # must build vocab with training data
            ##################################
            if build_vocab_flag:
                raise NotImplementedError("Are you sure?")
                # self.build_qa_vocab()
            self.load_qa_vocab()
        else:
            assert split in ['val_unbiased', 'testdev', 'val_all']
            ##################################
            # load qa vocab
            ##################################
            if load_vocab_flag:
                self.load_qa_vocab()

        print("finished loading the data, totally {} instances".format(len(self.data)))
        self.vocab = GQATorchDataset.TEXT.vocab

    def __getitem__(self, index):

        ##################################
        # Unpack GQA data items.
        # - Scene graph would be handled seperately
        ##################################
        datum = self.data[index]

        imageId = datum[0]
        question_text = datum[1]
        # new_programs = datum[2] # processed format, not the origin program
        questionID = datum[3]
        short_answer = datum[4]
        full_answer_text = datum[5]
        # program_text_tokenized = datum[6]
        # annotations = datum[7] # dict keys = ["answer", "question", "fullAnswer"]
        new_execution_buffer = datum[8]
        new_programs_hierarchical_decoder = datum[9]
        types = datum[10]

        ##################################
        # Prepare short answer
        # if it is in plural form, convert it into singular form
        ##################################
        # TODO: Find the semantically nearest label to 'bottle cap' or extend the vocab
        #   Problem: there is one label in testdev set, 'bottle cap', that is OOD
        #   Quick fix: replace 'bottle cap' with a random short answer label
        if short_answer == 'bottle cap':
            short_answer = 'bottle'
        assert short_answer in self.ans2label, short_answer
        # if short_answer in self.answer_plural_inv_map:
        #     short_answer = self.answer_plural_inv_map[short_answer]
        short_answer_label = self.ans2label[short_answer]

        ##################################
        # Pre-processing (tokenization only), relying on collate_fn to pad and numericalize
        ##################################
        question_text_tokenized = GQATorchDataset.TEXT.preprocess(question_text)
        full_answer_text_tokenized = GQATorchDataset.TEXT.preprocess(full_answer_text)

        ##################################
        # handle scene graph part
        ##################################
        queryID = str(imageId)
        # sg_datum = self.sg_feature_lookup.query(queryID)
        sg_datum = self.sg_feature_lookup.query_and_translate(queryID, new_execution_buffer) #, fetched_datum=datum)

        ##################################
        # Do padding
        ##################################
        # cut-off
        new_programs_hierarchical_decoder = new_programs_hierarchical_decoder[: GQATorchDataset.MAX_EXECUTION_STEP]
        # padding empty
        new_programs_hierarchical_decoder += (GQATorchDataset.MAX_EXECUTION_STEP - len(new_programs_hierarchical_decoder)) * [[]]
        # assign new
        program_text_tokenized = new_programs_hierarchical_decoder

        ##################################
        # return
        ##################################

        return (
            questionID, question_text_tokenized, sg_datum, program_text_tokenized,
            full_answer_text_tokenized, short_answer_label, types
        )

    def __len__(self):
        return len(self.data)

    @property
    def num_answers(self):
        return len(self.ans2label)

    def build_qa_vocab(self):
        ##################################
        # construct vocab using torchtext, one vocab for Question, Program, Full Answer
        ##################################
        tmp_text_list = []
        for datum in self.data:
            question_text = datum[1]
            program_text_tokenized = datum[6]
            full_answer_text = datum[5]
            # Question
            # must do preprocess (tokenization) before doing the buld vocab
            question_text_tokenized = GQATorchDataset.TEXT.preprocess(question_text)
            tmp_text_list.append(question_text_tokenized)
            # Program
            tmp_text_list.append(program_text_tokenized)
            # Full Answer
            # must do preprocess (tokenization) before doing the buld vocab
            full_answer_text_tokenized = GQATorchDataset.TEXT.preprocess(full_answer_text)
            tmp_text_list.append(full_answer_text_tokenized)
        GQATorchDataset.TEXT.build_vocab(tmp_text_list, vectors="glove.6B.300d")
        del tmp_text_list

        # save the vocab
        with open(ROOT_DIR / 'GraphVQA' / 'questions' / 'GQA_TEXT_obj.pkl', 'wb') as f:
            pickle.dump(GQATorchDataset.TEXT, f)

    def load_qa_vocab(self):
        ##################################
        # load existing vocab using pickle
        ##################################
        with open(ROOT_DIR / 'GraphVQA' / 'questions' / 'GQA_TEXT_obj.pkl', 'rb') as f:
            text_reloaded = pickle.load(f)
            GQATorchDataset.TEXT = text_reloaded

    @classmethod
    def indices_to_string(cls, indices, words=False):
        """Convert word indices (torch.Tensor) to sentence (string).
        Args:
            indices: torch.tensor or numpy.array of shape (T) or (T, 1)
            words: boolean, wheter return list of words
        Returns:
            sentence: string type of converted sentence
            words: (optional) list[string] type of words list
        """
        sentence = list()
        for idx in indices:
            word = GQATorchDataset.TEXT.vocab.itos[idx.item()]

            if word in ["<pad>", "<start>"]:
                continue
            if word in ["<end>"]:
                break

            # no needs of space between the special symbols
            if len(sentence) and word in ["'", ".", "?", "!", ","]:
                sentence[-1] += word
            else:
                sentence.append(word)

        if words:
            return " ".join(sentence), sentence
        return " ".join(sentence)


"""Example json input from QA
"07333408": {
    "semantic": [{"operation": "select", "dependencies": [], "argument": "wall (722332)"},
                 {"operation": "filter color", "dependencies": [0], "argument": "white"},
                 {"operation": "relate", "dependencies": [1], "argument": "_,on,s (722335)"},
                 {"operation": "query", "dependencies": [2], "argument": "name"}], "entailed": [], "equivalent": ["07333408"],
    "question": "What is on the white wall?",
    "imageId": "2375429",
    "isBalanced": true,
    "groups": {"global": "", "local": "14-wall_on,s"},
    "answer": "pipe",
    "semanticStr": "select: wall (722332)->filter color: white [0]->relate: _,on,s (722335) [1]->query: name [2]",
    "annotations": {"answer": {"0": "722335"},
    "question": {"4:6": "722332"},
    "fullAnswer": {"1": "722335", "5": "722332"}},
    "types": {"detailed": "relS", "semantic": "rel", "structural": "query"},
    "fullAnswer": "The pipe is on the wall."
    },
"""


def GQATorchDataset_collate_fn(data):
    """
    The collate_fn function working with pytorch data loader.

    Since we have both torchtext data and pytorch geometric data, and both of them
    require non-defualt data batching behaviour, we therefore create a custom collate fn funciton.

    input: data: is a list of tuples with (question_src, sg_datum, program_trg, ...)
            which is determined by the get item method of the dataset

    output: all fileds in batched format
    """
    questionID, question_text_tokenized, sg_datum, \
        program_text_tokenized, full_answer_text_tokenized, short_answer_label, types \
        = zip(*data)

    # question_text_tokenized, sg_datum, program_text_tokenized, full_answer_text_tokenized = zip(*data)

    # print("question_text_tokenized", question_text_tokenized)
    question_text_processed = GQATorchDataset.TEXT.process(question_text_tokenized)
    # print("question_text_processed", question_text_processed)

    # print("sg_datum", sg_datum)
    sg_datum_processed = torch_geometric.data.Batch.from_data_list(sg_datum)
    # print("sg_datum_processed", sg_datum_processed)

    # print("program_text_tokenized", program_text_tokenized)
    # program_text_tokenized_processed = GQATorchDataset.TEXT.process(program_text_tokenized)
    expand_program_text_tokenized = []
    for instrs in program_text_tokenized:
        expand_program_text_tokenized.extend(instrs)
    assert len(expand_program_text_tokenized) % GQATorchDataset.MAX_EXECUTION_STEP == 0, expand_program_text_tokenized
    program_text_tokenized_processed = GQATorchDataset.TEXT.process(expand_program_text_tokenized)
    # print("program_text_tokenized_processed", program_text_tokenized_processed)

    # print("full_answer_text_tokenized", full_answer_text_tokenized)
    full_answer_text_tokenized_processed = GQATorchDataset.TEXT.process(full_answer_text_tokenized)
    # print("full_answer_text_tokenized_processed", full_answer_text_tokenized_processed)

    short_answer_label = torch.LongTensor(short_answer_label)

    return (
        questionID, question_text_processed, sg_datum_processed, program_text_tokenized_processed,
        full_answer_text_tokenized_processed, short_answer_label, types
    )

if __name__ == '__main__':

    torch_geometric.set_debug(True)
    # split = 'train_unbiased'
    split = 'val_unbiased'
    # split = 'testdev'

    dataset = GQATorchDataset(split, build_vocab_flag=False, load_vocab_flag=True)
    # dataset[0]

    ##################################
    # Make sure the matcher results are correct
    # Double check the warning
    ##################################
    # sg_datum = dataset.sg_feature_lookup.query_and_translate(queryID='2361976', new_execution_buffer=[])


    # Set shuffle to True so that we could see different samples each time. showing a small sample
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=GQATorchDataset_collate_fn
    )

    # large batching with more workers for faster iterations
    # data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=512,
    #     shuffle=True,
    #     num_workers=32,
    #     collate_fn=GQATorchDataset_collate_fn
    # )

    for data_batch in data_loader:
        questionID, questions, gt_scene_graphs, programs, full_answers, short_answer_label, types = data_batch
        print("gt_scene_graphs", gt_scene_graphs)
        print("gt_scene_graphs.x", gt_scene_graphs.x)
        print("gt_scene_graphs.edge_attr", gt_scene_graphs.edge_attr )
        print("gt_scene_graphs.y", gt_scene_graphs.y)
        print("gt_scene_graphs.added_sym_edge", gt_scene_graphs.added_sym_edge)
        this_batch_size = questions.size(1)
        for batch_idx in range(min(this_batch_size, 128)):
            question = questions[:, batch_idx].cpu()
            question_sent, _ = GQATorchDataset.indices_to_string(question, True)
            print("question_sent", question_sent)

            for instr_idx in range(GQATorchDataset.MAX_EXECUTION_STEP):
                true_batch_idx = instr_idx + GQATorchDataset.MAX_EXECUTION_STEP * batch_idx
                gt = programs[:, true_batch_idx].cpu()
                gt_sent, _ = GQATorchDataset.indices_to_string(gt, True)
                if len(gt_sent) > 0:
                    print("gt_sent", gt_sent)

        ##################################
        # Only for ground truth scene graph
        # Using the scene graph vocab to decode the scene graph into text format
        ##################################
        if isinstance(dataset.sg_feature_lookup, GQA_gt_sg_feature_lookup):
            # for idx in range(gt_scene_graphs.num_edges):
            #     edge_str = GQA_gt_sg_feature_lookup.SG_ENCODING_TEXT.vocab.itos[
            #         gt_scene_graphs.edge_attr[idx, 0].item()
            #     ]
            #     print("edge_str", edge_str)
            for idx in range(gt_scene_graphs.num_nodes):
                for j in range(gt_scene_graphs.num_node_features):
                    obj_str = gt_scene_graphs.x[idx, j].item()
                    obj_str = GQA_gt_sg_feature_lookup.SG_ENCODING_TEXT.vocab.itos[obj_str]
                    print(obj_str, end=', ')
                print()
            
            print(gt_scene_graphs.y)
        break
