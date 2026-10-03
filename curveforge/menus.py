# SPDX-License-Identifier: GPL-3.0-or-later
"""Add menu of the CurveForge node editor (Shift+A)."""

import nodeitems_utils
from nodeitems_utils import NodeCategory, NodeItem

from .nodes import TREE_ID


class CFCategory(NodeCategory):
    @classmethod
    def poll(cls, context):
        return getattr(context.space_data, "tree_type", "") == TREE_ID


CATEGORIES = [
    CFCategory("CF_INPUT", "Input", items=[
        NodeItem("CF_NodeSpline"), NodeItem("CF_NodeSegment"), NodeItem("CF_NodeGap")]),
    CFCategory("CF_GENERATOR", "Generator", items=[NodeItem("CF_NodeLinear")]),
    CFCategory("CF_OPERATOR", "Operator", items=[
        NodeItem("CF_NodeCompose"), NodeItem("CF_NodeSequence"), NodeItem("CF_NodeRandomize"),
        NodeItem("CF_NodeConditional"), NodeItem("CF_NodeSelector"), NodeItem("CF_NodeMirror"),
        NodeItem("CF_NodeTransform"), NodeItem("CF_NodeMaterial"), NodeItem("CF_NodeUVTransform")]),
    CFCategory("CF_NUMBER", "Number", items=[
        NodeItem("CF_NodeValue"), NodeItem("CF_NodeInteger"), NodeItem("CF_NodeInfo"), NodeItem("CF_NodeMath"),
        NodeItem("CF_NodeRandom"), NodeItem("CF_NodeExpression"), NodeItem("CF_NodeCombine")]),
    CFCategory("CF_LAYOUT", "Layout", items=[NodeItem("NodeFrame"), NodeItem("NodeReroute")]),
]


def register():
    nodeitems_utils.register_node_categories("CURVEFORGE_NODES", CATEGORIES)


def unregister():
    nodeitems_utils.unregister_node_categories("CURVEFORGE_NODES")
