from typing import *
from collections import deque, defaultdict, Counter
import heapq
import bisect

# =========================
# Data structure definitions
# =========================

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return f"ListNode({self.val})"


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return f"TreeNode({self.val})"


# =========================
# Linked list helpers
# =========================

def build_linked_list(values: List[int]) -> Optional[ListNode]:
    if not values:
        return None

    dummy = ListNode()
    cur = dummy
    for v in values:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next


def linked_list_to_list(head: Optional[ListNode]) -> List[int]:
    res = []
    while head:
        res.append(head.val)
        head = head.next
    return res


# =========================
# Tree helpers
# =========================

def build_tree(values: List[Optional[int]]) -> Optional[TreeNode]:
    if not values or values[0] is None:
        return None

    root = TreeNode(values[0])
    q = deque([root])
    i = 1

    while q and i < len(values):
        node = q.popleft()

        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            q.append(node.left)
        i += 1

        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            q.append(node.right)
        i += 1

    return root


def tree_to_list(root: Optional[TreeNode]) -> List[Optional[int]]:
    if not root:
        return []

    res = []
    q = deque([root])

    while q:
        node = q.popleft()
        if node is None:
            res.append(None)
        else:
            res.append(node.val)
            q.append(node.left)
            q.append(node.right)

    while res and res[-1] is None:
        res.pop()

    return res


# =========================
# Output formatter
# =========================

def format_output(ans):
    if isinstance(ans, ListNode):
        return linked_list_to_list(ans)
    if isinstance(ans, TreeNode):
        return tree_to_list(ans)
    return ans


# =========================
# Solution
# =========================

class Solution:
    # paste the leetcode function here
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # in order: left, root, right
        # if not root:
        #     return []
        # return self.inorderTraversal(root.left) + root.val + self.inorderTraversal(root.right)

        # traversal loop
        # iterative (stack) solution
        res = []
        stack = []
        
        # root is where you're currently exploring
        # stack: places need to come back to
        while root or stack:
            # travel to as left as you can
            while root:
                stack.append(root)
                root = root.left

            # at this point root = none
            # no more nodes to travel left anymore
            root = stack.pop()  # move 1 step backward
            res.append(root.val) # add left element
            root = root.right 

        return res

# =========================
# Runner
# =========================

if __name__ == "__main__":
    sol = Solution()

    method_name = [name for name in dir(sol) if not name.startswith("__")][0]
    method = getattr(sol, method_name)

    args = [
        # argument examples:
        # array/string problems: [2, 7, 11, 15], 9
        # linked list problems: build_linked_list([1,2,3])
        # tree problems: build_tree([1, None, 2, 3])
        build_tree([1,2,3,4,5,None,8,None,None,6,7,9])
    ]

    ans = method(*args)
    print(format_output(ans))