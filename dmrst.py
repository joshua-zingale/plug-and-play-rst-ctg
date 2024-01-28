#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import sys
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModel
from DMRST_Parser_main.model_depth_edited import ParsingNet
from DMRST_Parser_main.MUL_main_Infer_edited import inference
from DMRST_Parser_main.DataHandler import get_RelationAndNucleus


# In[ ]:


class DMRST():
    def __init__(self, device = "cuda:0"):
        
        if device == "cpu":
            pass
        else:
            torch.cuda.set_device(device)
        
        model_path = sys.path[0] + "/DMRST_Parser_main/depth_mode/Savings/multi_all_checkpoint.torchsave"
        self.batch_size = 1
        save_path = "./model-cache/"

        self.bert_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
        bert_model = AutoModel.from_pretrained("xlm-roberta-base")

        bert_model = bert_model.to(device)

        for name, param in bert_model.named_parameters():
            param.requires_grad = False

        model = ParsingNet(bert_model, bert_tokenizer=self.bert_tokenizer, device = device)

        model = model.to(device)
        model.load_state_dict(torch.load(model_path, map_location = torch.device(device)))
        self.model = model.eval()
        
        
        self.relation_to_index_table = {'Attribution_SN': 0,
                                         'Enablement_NS': 1,
                                         'Cause_SN': 2,
                                         'Cause_NN': 3,
                                         'Temporal_SN': 4,
                                         'Condition_NN': 5,
                                         'Cause_NS': 6,
                                         'Elaboration_NS': 7,
                                         'Background_NS': 8,
                                         'Topic-Comment_SN': 9,
                                         'Elaboration_SN': 10,
                                         'Evaluation_SN': 11,
                                         'Explanation_NN': 12,
                                         'TextualOrganization_NN': 13,
                                         'Background_SN': 14,
                                         'Contrast_NN': 15,
                                         'Evaluation_NS': 16,
                                         'Topic-Comment_NN': 17,
                                         'Condition_NS': 18,
                                         'Comparison_NS': 19,
                                         'Explanation_SN': 20,
                                         'Contrast_NS': 21,
                                         'Comparison_SN': 22,
                                         'Condition_SN': 23,
                                         'Summary_SN': 24,
                                         'Explanation_NS': 25,
                                         'Enablement_SN': 26,
                                         'Temporal_NN': 27,
                                         'Temporal_NS': 28,
                                         'Topic-Comment_NS': 29,
                                         'Manner-Means_NS': 30,
                                         'Same-Unit_NN': 31,
                                         'Summary_NS': 32,
                                         'Contrast_SN': 33,
                                         'Attribution_NS': 34,
                                         'Manner-Means_SN': 35,
                                         'Joint_NN': 36,
                                         'Comparison_NN': 37,
                                         'Evaluation_NN': 38,
                                         'Topic-Change_NN': 39,
                                         'Topic-Change_NS': 40,
                                         'Summary_NN': 41}
        
        self.index_to_relation_table = {
            value: key for key, value in self.relation_to_index_table.items()
        }
    
    def index_to_relation(self, idx) -> str:
        '''Given a valid index returned by the logits of the inference model, returns the
        corresponding relation classification
        
        :return: The relation & nuclearity pair in form "Relation_NN", where
            the nuclearity is two characters, "NN", "NS", or "SN". Returns None if invalid index'''
        return self.index_to_relation_table.get(idx)
    
    def relation_to_index(self, relation):
        '''Given a valid relation, returns the corresponding relation index.
        
        :param relation: a string formatted "Relation_NN"
            
        :return: the index if the relation is valid. If the relation and nuclearity combination
            does not exist, returns None
        '''
        
        return self.relation_to_index_table.get(relation)
        
    
    def infer(self, input_sequences, input_EDU_breaks = None):
        return inference(self.model, self.bert_tokenizer, input_sequences, self.batch_size, input_EDU_breaks)
    


# In[84]:


class RSTNode:
    SATELLITE = "Satellite"
    NUCLEUS = "Nucleus"
    ROOT = "Root" # The nuclearity is Root for the root of a tree
    LEAF = "Leaf" # The relation is Leaf for leafs
    
    def __init__(self, lower, upper, nuclearity, relation = None):
        self.relation = relation
        self.nuclearity = nuclearity
        self.lower = lower
        self.upper = upper
        self.left = None
        self.right = None
        self.parent = None
        
        
    def get_text(self, tokens, EDU_breaks):
        '''The tokens output from the inference function of DMRST alongside the
        EDU_breaks in the same output. Will turn all underscores into spaces'''
        
        EDU_breaks = [-1] + EDU_breaks
        
        low_idx = EDU_breaks[self.lower - 1] + 1
        high_idx = EDU_breaks[self.upper] + 1
        
        return ''.join(tokens[low_idx: high_idx]).replace("▁", " ")
        
    def __str__(self):
        return f"<{self.lower}-{self.upper}>"
    
    def __repr__(self):
        return self.__str__()

class RSTTree:
    def __init__(self, inference = None):
        '''Initializes an RSTTree.
        :param inference: the string output from the DMRST parser that given all of the relations.'''
        self.root = None
        self.leafs = [] # Will hold a pointer to each leaf node
        if inference:
            relations = inference.split(" ")
            self._build_from_relations(relations)
    
    def get_dependencies(self, leaf_node):
        '''Returns the dependencies of the passed in node. A node has a dependency on another node
        if, in a retorical dependency graph as defined in Xing et al. 2022, there exists a
        unidirectional path between the node and the other. For this function, all nuclei are used,
        not only the leftmost as was done in the paper.
        
        Xing et al., 2022, Discourse Parsing Enhanced by Discourse Dependence Perception
        '''
        
        dependencies = []
        if leaf_node.nuclearity == RSTNode.SATELLITE:
            relation = leaf_node.parent.relation
            heads = self._get_heads(leaf_node.parent)
            
            dependencies = list(map(lambda x: self._make_dependency_pair(leaf_node, x, relation), heads))
        
        if leaf_node.nuclearity == RSTNode.NUCLEUS:
            
            # Go up through the tree, finding relationships between leaf_node and others
            curr = leaf_node
            ancestor = curr.parent
            prev = leaf_node # To exit one after encountering a satellite
            while (curr.parent != None and prev.nuclearity != RSTNode.SATELLITE):
                
                ancestor = curr.parent
                sibling = ancestor.left if ancestor.right is curr else ancestor.right  
                relation = ancestor.relation
                
                heads = self._get_heads(sibling)
                if leaf_node in heads: heads.remove(leaf_node) # no dependency with itself
                dependencies.extend(list(map(
                    lambda x: self._make_dependency_pair(leaf_node, x, relation),
                    heads)))
                
                prev = curr
                curr = ancestor
                
        return dependencies
        
    
        
    def _make_dependency_pair(self, leaf_node1, leaf_node2, relation):
        '''Given a two leaf nodes and a relation, returns a dependency pair,
        ie. (leaf_i, leaf_j, relation) where i < j.'''
        
        if leaf_node1.lower < leaf_node2.lower:
            return (leaf_node1, leaf_node2, relation)
        else:
            return (leaf_node2, leaf_node1, relation)
            
    def _get_heads(self, internal_node):
        '''From an internal node, follows child-paths only along edges to nuclei.
        If internal_node is actually a leaf node, returns a list with itself'''
      
        heads = []
        self._get_heads_help(internal_node, heads)
        
        return heads
    
    def _get_heads_help(self, internal_node, heads):
        
        if internal_node.relation == RSTNode.LEAF:
            heads.append(internal_node)
            return heads
        
        if internal_node.left.nuclearity == RSTNode.NUCLEUS:
                if internal_node.left.relation == RSTNode.LEAF:
                    heads.append(internal_node.left)
                else:
                    self._get_heads_help(internal_node.left, heads)
                    
        if internal_node.right.nuclearity == RSTNode.NUCLEUS:
                if internal_node.right.relation == RSTNode.LEAF:
                    heads.append(internal_node.right)
                else:
                    self._get_heads_help(internal_node.right, heads)
                    
        return heads
                    
    def _build_from_relations(self, relations, subtree_root = None):
        l_l, n_l, r_l, u_l, l_r, n_r, r_r, u_r = self._info_from_string(relations[0])
        
        # What relation the current node will hold
        relation = r_r
        if n_l == RSTNode.SATELLITE:
            relation = r_l
            
        # create root node
        if subtree_root == None:
            root = RSTNode(l_l, u_r, RSTNode.ROOT, relation)
            self.insert(root)
            subtree_root = root
        else: # internal node
            subtree_root.relation = relation
                
        # create and append left & right nodes
        left = RSTNode(l_l, u_l, n_l)
        right = RSTNode(l_r, u_r, n_r)
        
        self.insert(left, subtree_root, insert_left = True)
        self.insert(right, subtree_root, insert_left = False)
        
        if l_l == u_l:
            # this will be a leaf node
            left.relation = RSTNode.LEAF
            self.leafs.append(left)
        else:
            # this will be an internal node
            relations.pop(0)
            self._build_from_relations(relations, subtree_root = left)
            
        if l_r == u_r:
            # this will be a leaf node
            right.relation = RSTNode.LEAF
            self.leafs.append(right)
        else:
            # this will be an internal node
            relations.pop(0)
            self._build_from_relations(relations, subtree_root = right)
            
        # add nuclearity information to relation
        subtree_root.relation += "_" + subtree_root.left.nuclearity[0] + subtree_root.right.nuclearity[0] 
            
            
    def _info_from_string(self, string):
        '''returns lower_left, nuclearity_left, relation_left, upper_left, lower_right,
        nuclearity_right, relation_right, upper_right'''
        spl = re.split("\(|:|=|,|\)", string)
        
        l_l, n_l, r_l, u_l = spl[1:5]
        
        l_r, n_r, r_r, u_r = spl[5:-1]
        
        return int(l_l), n_l, r_l, int(u_l), int(l_r), n_r, r_r, int(u_r )
        

    def insert(self, node, parent=None, insert_left=True):
        new_node = node
        if self.root is None:
            self.root = new_node
        else:
            if parent is None:
                raise ValueError("parent cannot be None when tree is not empty.")
            self._insert_at_node(parent, new_node, insert_left)
            
        return new_node

    def _insert_at_node(self, parent_node, new_node, insert_left):
        if insert_left:
            if parent_node.left is None:
                parent_node.left = new_node
            else:
                raise ValueError("Node already has a left child.")
        else:
            if parent_node.right is None:
                parent_node.right = new_node
            else:
                raise ValueError("Node already has a right child.")

        new_node.parent = parent_node  # Set the parent pointer of the new node


    def inorder_traversal(self, node):
        if node:
            self.inorder_traversal(node.left)
            print("Relation:", node.relation, "Nuclearity:", node.nuclearity, "Lower:", node.lower, "Upper:", node.upper)
            self.inorder_traversal(node.right)
            
      
    def get_leaf_nodes(self):
        leaf_nodes = []
        self._get_leaf_nodes_recursive(self.root, leaf_nodes)
        return leaf_nodes
            
    def _get_leaf_nodes_recursive(self, node, leaf_nodes):
        if node is None:
            return

        if node.left is None and node.right is None:
            leaf_nodes.append(node)

        self._get_leaf_nodes_recursive(node.left, leaf_nodes)
        self._get_leaf_nodes_recursive(node.right, leaf_nodes)