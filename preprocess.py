"""RUN
python preprocess.py create_balanced_programs

For list mapping table, Search for code:

symbolic_programs.append((

"""

import re
import json
import sys
import pathlib
import time
import nltk
from nltk.stem import WordNetLemmatizer
import Constants
import numpy as np
import random
import argparse


ROOT_DIR = Constants.ROOT_DIR
# PACKAGE_DIR = ROOT_DIR / 'DialogGQA'
# GQA_DETR_OD_DIR = PACKAGE_DIR / 'GQAdetrObjectDetection-master'
# SCENEGRAPHS = ROOT_DIR / 'Downloads' / 'sceneGraphs'
# sys.path.insert(0, str(GQA_DETR_OD_DIR))

lemmatizer = WordNetLemmatizer()


def get_args_parser():
    parser = argparse.ArgumentParser('Explainable GQA Parser', add_help=False)

    parser.add_argument('--val-all', default=False, type=bool, metavar='val_all',
                        help='generate val-all programs json file')

    return parser


def add1(string, extra):
    #nums = string[1:-1].split(',')
    #nums = [str(int(_) + extra) for _ in nums]
    new_string = ""
    for c in string:
        if c.isdigit():
            new_string += str(int(c) + extra)
        else:
            new_string += c
    return new_string


def filter_field(string):
    output = re.search(r' ([^ ]+)\b', string).group()[2:]
    if 'not(' in output:
        return re.search(r'\(.+$', output).group()[1:], True
    else:
        return output, False


def filter_parenthesis(string):
    objects = re.search(r'\(.+\)', string).group()[1:-1]
    if objects == '-':
        return '[]', objects
    else:
        return '[{}]'.format(objects), objects

# def filter_parenthesis(string):
#     objects = re.search(r'\(.+\)', string).group()[1:-1]
#     if objects == '-':
#         return '[]'
#     else:
#         return '[{}]'.format(objects)


def filter_squre(string):
    indexes = re.search(r'\[.+\]', string).group()
    if ',' in indexes:
        return ','.join(['[{}]'.format(_.strip()) for _ in indexes[1:-1].split(',')])
    else:
        return indexes


def extract_rel(string):
    subject = re.search(r'^([^,]+),', string).group()[:-1]
    relation = re.search(r',(.+),', string).group()[1:-1]
    try:
        o_s = re.search(r',(o|s) ', string).group()[1:-1]
        if 's' in o_s:
            return subject, relation, True
        else:
            return subject, relation, False
    except:
        return subject, relation, None


def extract_query_key(string):
    if 'name' in string:
        return 'name'
    elif 'hposition' in string:
        return 'hposition'
    elif 'vposition' in string:
        return 'vposition'
    else:
        return 'attributes'


def split_rel(string):
    subject = re.search(r'([^,]+),', string).group()[:-1]
    relation1 = re.search(r',(.+)\|', string).group()[1:-1]
    relation2 = re.search(r'\|(.+),', string).group()[1:-1]
    o_s = re.search(r',(o|s)', string).group()[1:-1]
    if 's' in o_s:
        return subject, relation1, relation2, True
    else:
        return subject, relation1, relation2, False


def split_attr(string):
    attr1 = re.search(r'(.+)\|', string).group()[2:-1]
    attr2 = re.search(r'\|(.+) ', string).group()[1:-1]
    return attr1, attr2


def shuffle(string):
    attrs = string.split('|')
    random.shuffle(attrs)
    attr1, attr2 = attrs
    return attr1, attr2


"""
Parse one line
"""
def split_in_generate_pairs(string):
    output = []
    buf_str = ""
    for s in string:
        if s == "(":
            string = buf_str.strip()
            if string:
                output.append(string)
            output.append("(")
            buf_str = ""
        elif s == ")":
            string = buf_str.strip()
            if string:
                output.append(string)
            output.append(")")
            buf_str = ""
        elif s == ",":
            string = buf_str.strip()
            if string:
                output.append(string)
            output.append(",")
            buf_str = ""
        else:
            buf_str += s
    return output

def generate_pairs(entry):
    if entry:
        output = []
        for r in entry:
            _, p = r.split('=')
            sub_p = split_in_generate_pairs(p)
            output.extend(sub_p)
            output.append(";")
        del output[-1]
    else:
        output = []
    return output


def generate_hierarchical_pairs(entry):
    if entry:
        output = []
        for r in entry:
            _, p = r.split('=')
            sub_p = split_in_generate_pairs(p)
            output.append(sub_p)
    else:
        output = []
        raise NotImplementedError
    return output

def preprocess(raw_data, output_path, dataset_this=None, sg_data=None):
    symbolic_programs = []
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    keys = list(raw_data.keys())
    print("total {} programs".format(len(keys)))
    success, fail = 0, 0

    empty_buffer_counter = 0
    multi_buffer_counter = 0
    total_buffer_counter = 0
    multi_2_buffer_count = 0
    max_full_anwer_len = 0
    max_instr_len = 0
    max_new_programs_decoder_len = 0


    for idx in range(len(keys)):
        imageId = raw_data[keys[idx]]['imageId']

        ##################################
        # Only get this files
        ##################################
        if dataset_this is None:
            pass
        else:
            if str(imageId) + '.jpg' not in dataset_this.filenames:
                # skip this
                continue
            else:
                pass

        question = raw_data[keys[idx]]['question']
        program = raw_data[keys[idx]]['semantic']
        answer = raw_data[keys[idx]]['answer']

        annotations = raw_data[keys[idx]]['annotations']
        fullAnswer = raw_data[keys[idx]]['fullAnswer']


        new_programs = []

        execution_buffer = []
        # try:
        for i, prog in enumerate(program):

            ##################################
            # Very important to understand the dependencies
            ##################################

            if prog['dependencies']:
                subject = ",".join(["[{}]".format(_) for _ in prog['dependencies']])

            if '(' in prog['argument'] and ')' in prog['argument'] and 'not(' not in prog['argument']:
                result, objects_str = filter_parenthesis(prog['argument'])
                # a str. one or multiple resutls.
                if objects_str != '-':
                    objects_str_list = objects_str.split(',')
                    objects_list = [ int(s) for s in objects_str_list ]
                else:
                    objects_list = []
                assert len(execution_buffer) == i
                execution_buffer.append(objects_list)
            else:
                result = '?'
                objects_list = []
                for de in prog['dependencies']:
                    objects_list.extend(execution_buffer[de])

                # if len(prog['dependencies']) == 0 and 'scene' not in str(program):
                #     print("program", program, "prog", prog, "objects_list", objects_list)
                assert len(execution_buffer) == i
                execution_buffer.append(objects_list)
                # raise NotImplementedError


            if prog['operation'] == 'select':
                if prog['argument'] == 'scene':
                    new_programs.append('{}=scene()'.format(result))
                    flag = 'full'
                else:
                    new_programs.append('{}=select({})'.format(
                        result, lemmatizer.lemmatize(prog['argument'].split(' ')[0])))
                    flag = 'partial'

            elif prog['operation'] == 'relate':
                # print prog['argument']
                name, relation, reverse = extract_rel(prog['argument'])
                if reverse == None:
                    new_programs.append('{}=relate_attr({}, {}, {})'.format(result, subject, relation, name))
                else:
                    if reverse:
                        if name != '_':
                            name = lemmatizer.lemmatize(name)
                            new_programs.append('{}=relate_inv_name({}, {}, {})'.format(
                                result, subject, relation, name))
                        else:
                            new_programs.append('{}=relate_inv({}, {})'.format(result, subject, relation))
                    else:
                        if name != '_':
                            name = lemmatizer.lemmatize(name)
                            new_programs.append('{}=relate_name({}, {}, {})'.format(result, subject, relation, name))
                        else:
                            new_programs.append('{}=relate({}, {})'.format(result, subject, relation))

            elif prog['operation'].startswith('query'):
                if prog['argument'] == "hposition":
                    new_programs.append('{}=query_h({})'.format(result, subject))
                elif prog['argument'] == "vposition":
                    new_programs.append('{}=query_v({})'.format(result, subject))

                elif prog['argument'] == "name":
                    new_programs.append('{}=query_n({})'.format(result, subject))
                else:
                    if flag == 'full':
                        new_programs.append('{}=query_f({})'.format(result, prog['argument']))
                    else:
                        new_programs.append('{}=query({}, {})'.format(result, subject, prog['argument']))

            elif prog['operation'] == 'exist':
                new_programs.append('{}=exist({})'.format(result, subject))

            elif prog['operation'] == 'or':
                new_programs.append('{}=or({})'.format(result, subject))

            elif prog['operation'] == 'and':
                new_programs.append('{}=and({})'.format(result, subject))

            elif prog['operation'].startswith('filter'):
                if prog['operation'] == 'filter hposition':
                    new_programs.append('{}=filter_h({}, {})'.format(result, subject, prog['argument']))

                elif prog['operation'] == 'filter vposition':
                    new_programs.append('{}=filter_h({}, {})'.format(result, subject, prog['argument']))

                else:
                    negative = 'not(' in prog['argument']
                    if negative:
                        new_programs.append('{}=filter_not({}, {})'.format(result, subject, prog['argument'][4:-1]))
                    else:
                        new_programs.append('{}=filter({}, {})'.format(result, subject, prog['argument']))

            elif prog['operation'].startswith('verify'):
                if prog['operation'] == 'verify':
                    new_programs.append('{}=verify({}, {})'.format(result, subject, prog['argument']))
                elif prog['operation'] == 'verify hposition':
                    new_programs.append('{}=verify_h({}, {})'.format(result, subject, prog['argument']))
                elif prog['operation'] == 'verify vposition':
                    new_programs.append('{}=verify_v({}, {})'.format(result, subject, prog['argument']))
                elif prog['operation'] == 'verify rel':
                    name, relation, reverse = extract_rel(prog['argument'])
                    name = lemmatizer.lemmatize(name)
                    if reverse:
                        new_programs.append('{}=verify_rel_inv({}, {}, {})'.format(result, subject, relation, name))
                    else:
                        new_programs.append('{}=verify_rel({}, {}, {})'.format(result, subject, relation, name))
                    # if reverse:
                    #    new_programs.append('?=relate_inv_name({}, {}, {})'.format(subject, relation, name))
                    #    new_programs.append('{}=exist([{}])'.format(result, len(new_programs) - 1))
                    # else:
                    #    new_programs.append('?=relate_name({}, {}, {})'.format(subject, relation, name))
                    #    new_programs.append('{}=exist([{}])'.format(result, len(new_programs) - 1))
                else:
                    if flag == 'full':
                        new_programs.append('{}=verify_f({})'.format(result, prog['argument']))
                    else:
                        new_programs.append('{}=verify({}, {})'.format(result, subject, prog['argument']))

            elif prog['operation'].startswith('choose'):
                if prog['operation'] == 'choose':
                    attr1, attr2 = shuffle(prog['argument'])
                    if flag == "full":
                        new_programs.append('{}=choose_f({}, {})'.format(result, attr1, attr2))
                    else:
                        new_programs.append('{}=choose({}, {}, {})'.format(result, subject, attr1, attr2))

                elif prog['operation'] == 'choose rel':
                    name, relation1, relation2, reverse = split_rel(prog['argument'])
                    relation1, relation2 = shuffle('{}|{}'.format(relation1, relation2))
                    name = lemmatizer.lemmatize(name)
                    if reverse:
                        new_programs.append('{}=choose_rel({}, {}, {}, {})'.format(
                            result, subject, name, relation1, relation2))
                    else:
                        new_programs.append('{}=choose_rel_inv({}, {}, {}, {})'.format(
                            result, subject, name, relation1, relation2))

                elif prog['operation'] == 'choose hposition':
                    attr1, attr2 = shuffle(prog['argument'])
                    new_programs.append('{}=choose_h({}, {}, {})'.format(result, subject, attr1, attr2))

                elif prog['operation'] == 'choose vposition':
                    attr1, attr2 = shuffle(prog['argument'])
                    new_programs.append('{}=choose_v({}, {}, {})'.format(result, subject, attr1, attr2))

                elif prog['operation'] == 'choose name':
                    attr1, attr2 = shuffle(prog['argument'])
                    attr1 = lemmatizer.lemmatize(attr1)
                    attr2 = lemmatizer.lemmatize(attr2)
                    new_programs.append('{}=choose_n({}, {}, {})'.format(result, subject, attr1, attr2))

                elif ' ' in prog['operation']:
                    attr = prog['operation'].split(' ')[1]
                    if len(prog['argument']) == 0:
                        new_programs.append('{}=choose_subj({}, {})'.format(result, subject, attr))
                    else:
                        attr1, attr2 = shuffle(prog['argument'])
                        if flag == "full":
                            new_programs.append('{}=choose_f({}, {})'.format(result, attr1, attr2))
                        else:
                            new_programs.append('{}=choose_attr({}, {}, {}, {})'.format(
                                result, subject, attr, attr1, attr2))

            elif prog['operation'].startswith('different'):
                if ' ' in prog['operation']:
                    attr = prog['operation'].split(' ')[1]
                    new_programs.append('{}=different_attr({}, {})'.format(result, subject, attr))
                else:
                    new_programs.append('{}=different({})'.format(result, subject))

            elif prog['operation'].startswith('same'):
                if ' ' in prog['operation']:
                    attr = prog['operation'].split(' ')[1]
                    new_programs.append('{}=same_attr({}, {})'.format(result, subject, attr))
                else:
                    new_programs.append('{}=same({})'.format(result, subject))

            elif prog['operation'] == 'common':
                new_programs.append('{}=common({})'.format(result, subject))

            else:
                raise ValueError("Unseen Function {}".format(prog))
            # if answer == "yes":
            #    answer = True
            # elif answer == "no":
            #    answer = False
            # elif 'choose' in new_programs[-1]:
            #    _, _, arguments = parse_program(new_programs[-1])
            #    if answer not in arguments:
            #        import pdb
            #        pdb.set_trace()
            # elif answer == "right" and 'choose' in new_programs[-1]:
            #    answer = 'to the right of'
            # elif answer == "left" and 'choose' in new_programs[-1]:
            #    answer = 'to the left of'

        ##################################
        # Add new fields for step wise execution.
        ##################################

        new_programs_decoder = generate_pairs(new_programs)
        max_new_programs_decoder_len = max( max_new_programs_decoder_len, len(new_programs_decoder) )
        new_programs_hierarchical_decoder = generate_hierarchical_pairs(new_programs)
        assert len(new_programs) == len(execution_buffer),  str(new_programs) + str(execution_buffer) + str(program)
        assert len(new_programs_hierarchical_decoder) == len(execution_buffer), str(new_programs) + str(execution_buffer) + str(program)

        if sg_data is not None:

            ##################################
            # Prepare inverse mapping
            # Must be the same with gqa_dataset/gqa.py
            ##################################
            # apply additional transformation step for the
            sg_objects = sg_data[imageId]['objects']
            # Sort the keys to ensure object order is always the same
            sorted_oids = sorted(list(sg_objects.keys()))

            gt_classes_i = []
            oid_to_idx = {}
            for oid in sorted_oids:

                obj = sg_objects[oid]

                # Compute object GT bbox
                b = np.array([obj['x'], obj['y'], obj['w'], obj['h']])
                try:
                    assert np.all(b[:2] >= 0), (b, obj)  # sanity check
                    assert np.all(b[2:] > 0), (b, obj)  # no empty box
                except:
                    # print("Invalid object detected, imageId:", imageId,
                    # "obj", obj,
                    # )
                    continue  # skip objects with empty bboxes or negative values


                oid_to_idx[oid] = len(gt_classes_i)
                # if len(obj['relations']) > 0:
                #     no_objs_with_rels = False

                # Compute object GT class # dummy counting only
                # gt_class = classes_to_ind[obj['name']]
                gt_class = obj['name']
                gt_classes_i.append(gt_class)

            del sorted_oids, gt_classes_i


            ##################################
            # Mapping scene graph object id to local sorted object id
            ##################################
            new_execution_buffer = []
            for instr_idx in range(len(execution_buffer)):
                sg_obj_id_list = execution_buffer[instr_idx]
                local_obj_id_list = []
                for oid_idx in range(len(sg_obj_id_list)):
                    sg_obj_id_str = str(sg_obj_id_list[oid_idx])
                    # local_obj_id_list.append( sorted_oids.index( str(sg_obj_id_list[oid_idx])  ) ) # contains empty object!
                    if sg_obj_id_str in oid_to_idx:
                        local_obj_id = oid_to_idx[sg_obj_id_str]
                        local_obj_id_list.append(local_obj_id)
                    else:
                        print("EXE Buffer Referring Empty Object!",
                        "sg_obj_id_list", sg_obj_id_list,
                        "imageId", imageId,
                        "question", question,
                        )

                new_execution_buffer.append(local_obj_id_list)

                total_buffer_counter += 1
                if len(local_obj_id_list) == 0:
                    empty_buffer_counter +=1
                elif len(local_obj_id_list) >= 2:
                    multi_buffer_counter += 1

                if len(local_obj_id_list) == 2 or len(local_obj_id_list) == 3 or len(local_obj_id_list) == 4:
                    multi_2_buffer_count += 1
                    # ==2: 9840
                    # ==2 and ==3: 10044
                    # ==2 and ==3 and ==4: 10124
                    # Neural Execution Engine Annotations: empty_buffer_counter 51083 multi_buffer_counter 10272 total_buffer_counter 412192 multi_2_buffer_count 9840

            assert len(execution_buffer) == len(new_execution_buffer)

            ##################################
            # Mapping annotation object ids
            ##################################

            annotations_keys = list(annotations.keys()) # dict keys = ["answer", "question", "fullAnswer"]
            new_annotations = dict()
            for annotation_key in annotations_keys:
                new_annotations[annotation_key] = dict()
                for k, v in annotations[annotation_key].items():
                    # new_annotations[annotation_key][k] = sorted_oids.index( str(v) )
                    if str(v) in oid_to_idx:
                        local_obj_id = oid_to_idx[str(v)]
                        new_annotations[annotation_key][k] = local_obj_id
                    else:
                        print(
                            "Ptr Annotations Referring Empty Object!",
                            "annotations", annotations,
                            "imageId", imageId,
                            "question", question,
                        )

            ##################################
            # Get instr and full answer stats
            ##################################
            for instr_idx in range(len(execution_buffer)):
                max_instr_len = max(max_instr_len, len(new_programs_hierarchical_decoder[instr_idx]))
            max_full_anwer_len = max(max_full_anwer_len, len(fullAnswer.split()) )

        else:
            # testdev
            # set all to empty
            new_execution_buffer = []
            new_annotations = dict()
            # raise NotImplementedError

        # symbolic_programs.append((imageId, question, new_programs, keys[idx], answer))
        symbolic_programs.append((
            imageId, # 0
            question, # 1
            [], # new_programs, # 2
            keys[idx], # 3, question ID
            answer, # 4
            fullAnswer, # 5
            new_programs_decoder, # 6 list: tokenized str
            new_annotations, # 7 dict keys = ["answer", "question", "fullAnswer"]
            new_execution_buffer, # 8 execution results
            new_programs_hierarchical_decoder, # 9 instruction split results
            raw_data[keys[idx]]['types'], # 10 types
        ))

        success += 1

        # except Exception:
        #    print(program)
        #    fail += 1

        if idx % 10000 == 0:
            sys.stdout.write("finished {}/{} \r".format(success, fail))

    print(
        "Neural Execution Engine Annotations:",
        "empty_buffer_counter", empty_buffer_counter,
        "multi_buffer_counter", multi_buffer_counter,
        "total_buffer_counter", total_buffer_counter,
        "multi_2_buffer_count", multi_2_buffer_count,
        "max_full_anwer_len", max_full_anwer_len,
        "max_instr_len", max_instr_len,
        "max_new_programs_decoder_len", max_new_programs_decoder_len
    )

    print("finished {}/{}".format(success, fail))
    with open(output_path, 'w') as f:
        json.dump(symbolic_programs, f, indent=2)


# arg = sys.argv[1]
# if arg == 'create_balanced_programs': # Modified by WX
if True:
    parser = argparse.ArgumentParser('Explainable GQA training and evaluation script',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    print(args.val_all)
    
    with open(ROOT_DIR / 'GraphVQA/questions/original/testdev_balanced_questions.json') as f:
        # total 12578 programs
        raw_dev_data = json.load(f)
    preprocess(raw_dev_data, ROOT_DIR / 'GraphVQA/questions/testdev_balanced_programs.json')

    # from gqa_dataset.visual_genome import VG
    # dataset_this = VG(
    #     'test', '/home/ubuntu/GQA/sgg/data_path/GQA',
    #     num_val_im=-1,
    #     filter_duplicate_rels=True,
    #     min_graph_size=-1,
    #     max_graph_size=-1,
    #     filter_non_overlap=False
    # )
    dataset_this = None

    # fileStr = SCENEGRAPHS / "val_sceneGraphs.json"
    fileStr = ROOT_DIR / 'GraphVQA/sceneGraphs/val_sceneGraphs.json'
    with open(fileStr) as f:
        sg_data = json.load(f)
    val_questions_path = ROOT_DIR / 'GraphVQA/questions/original/val_balanced_questions.json'
    val_programs_path = ROOT_DIR / 'GraphVQA/questions/val_balanced_programs.json'
    # val_questions_path = ROOT_DIR / 'GraphVQA/questions/original/val_balanced_masked_questions.json'
    # val_programs_path = ROOT_DIR / 'GraphVQA/questions/val_balanced_masked_programs.json'
    with open(val_questions_path) as f:
        # total 132062 programs
        raw_data = json.load(f)
    # actual: 131548, discard 514 (0.3%)
    preprocess(raw_data, val_programs_path, dataset_this, sg_data)

    # exit(0)

    # with open('questions/original/train_balanced_questions.json') as f:
    #     # total 943000 programs
    #     raw_data = json.load(f)
    # preprocess(raw_data, 'questions/train_balanced_programs.json') # new

    # from gqa_dataset.visual_genome import VG
    # dataset_this = VG(
    #     'train',
    #     '/home/ubuntu/GQA/sgg/data_path/GQA',
    #     num_val_im=-1,
    #     filter_duplicate_rels=True,
    #     min_graph_size=-1,
    #     max_graph_size=-1,
    #     filter_non_overlap=False
    # )
    dataset_this = None 

    # fileStr = SCENEGRAPHS / "train_sceneGraphs.json"
    fileStr = ROOT_DIR / 'GraphVQA/sceneGraphs/train_sceneGraphs.json'
    with open(fileStr) as f:
        sg_data = json.load(f)
    train_questions_path = ROOT_DIR / 'GraphVQA/questions/original/train_balanced_questions.json'
    train_programs_path = ROOT_DIR / 'GraphVQA/questions/train_balanced_programs.json'
    # train_questions_path = ROOT_DIR / 'GraphVQA/questions/original/train_balanced_masked_questions.json'
    # train_programs_path = ROOT_DIR / 'GraphVQA/questions/train_balanced_masked_programs.json'
    with open(train_questions_path) as f:
        # total 943000 programs
        raw_data = json.load(f)

    # only 939806, discard 3194 (0.3%)
    preprocess(raw_data, train_programs_path, dataset_this, sg_data)

    # with open('questions/original/val_balanced_questions.json') as f:
    #     # total 132062 programs
    #     raw_data = json.load(f)
    # preprocess(raw_data, 'questions/val_balanced_programs.json') # new


    # with open('questions/original/val_balanced_questions.json') as f:
    #     raw_data = json.load(f)
    # with open('questions/original/val_balanced_questions.json') as f:
    #     raw_data.update(json.load(f))
    # preprocess(raw_data, 'questions/trainval_balanced_programs.json')

    if args.val_all:
        dataset_this = None

        # fileStr = SCENEGRAPHS / "val_sceneGraphs.json"
        fileStr = ROOT_DIR / 'explainableGQA/sceneGraphs/val_sceneGraphs.json'
        with open(fileStr) as f:
            sg_data = json.load(f)
        val_questions_path = ROOT_DIR / 'explainableGQA/questions/original/val_all_questions.json'
        val_programs_path = ROOT_DIR / 'explainableGQA/questions/val_all_programs.json'
        # val_questions_path = ROOT_DIR / 'explainableGQA/questions/original/val_balanced_masked_questions.json'
        # val_programs_path = ROOT_DIR / 'explainableGQA/questions/val_balanced_masked_programs.json'
        with open(val_questions_path) as f:
            # total 132062 programs
            raw_data = json.load(f)
        # actual: 131548, discard 514 (0.3%)
        # print(raw_data)
        preprocess(raw_data, val_programs_path, dataset_this, sg_data)
