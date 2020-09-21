---
layout:     post
title:      LeetCode递归题解
subtitle:   递归专题
date:       2019-11-15
author:     dhx20150812
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - Leetcode
    - Recursive
---


# 938. Range Sum of BST

> 题目链接见 https://leetcode.com/problems/range-sum-of-bst/

Given the `root` node of a binary search tree, return the sum of values of all nodes with value between L and R (inclusive).

The binary search tree is guaranteed to have unique values.

 **Example 1**:
 ```
Input: root = [10,5,15,3,7,null,18], L = 7, R = 15
Output: 32
```

**Example 2**：
```
Input: root = [10,5,15,3,7,13,18,1,null,6], L = 6, R = 10
Output: 23
```

**Note:**
```
1. The number of nodes in the tree is at most 10000.
2. The final answer is guaranteed to be less than 2^31.
```
**Solution** : 简单的递归（中序遍历）

```c++
class Solution {
public:
    int rangeSumBST(TreeNode* root, int L, int R) {
        int sum = 0;
        helper(root, L, R, sum);
        return sum;
    }
    void helper(TreeNode* root, int L, int R, int &sum){
        if(!root) return;
        if(root->val >= L && root->val <= R)
            sum += root->val;
        helper(root->left, L, R, sum);
        helper(root->right, L, R, sum);
    }
};
```

# 783. Minimum Distance Between BST Nodes

> 题目链接见 https://leetcode.com/problems/minimum-distance-between-bst-nodes/

Given a Binary Search Tree (BST) with the root node root, return the minimum difference between the values of **any two different nodes** in the tree.

**Example** :
```
Input: root = [4,2,6,1,3,null,null]
Output: 1
Explanation:
Note that root is a TreeNode object, not an array.

The given tree [4,2,6,1,3,null,null] is represented by the following diagram:

          4
        /   \
      2      6
     / \    
    1   3  

while the minimum difference in this tree is 1, it occurs between node 1 and node 2, also between node 3 and node 2.
```
**Note**:
```
The size of the BST will be between 2 and 100.
The BST is always valid, each node's value is an integer, and each node's value is different.
```

**注意是任意节点间的最小距离**

**Solution**: 简单的递归（中序遍历）
```c++
class Solution {
public:
    int minDiffInBST(TreeNode* root) {
        int res = INT_MAX, pre = -1;
        helper(root, res, pre);
        return res;
    }
    
    void helper(TreeNode *root, int &res, int &pre){
        if(root){
            helper(root->left, res, pre);
            if(pre >= 0)
                res = min(res, root->val - pre);
            pre = root->val;
            helper(root->right, res, pre);
        }
    }
};
```

# 17. Letter Combinations of a Phone Number

> 题目链接见：https://leetcode.com/problems/letter-combinations-of-a-phone-number/

Given a string containing digits from `2-9` inclusive, return all possible letter combinations that the number could represent.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

**Note:**

Although the above answer is in lexicographical order, your answer could be in any order you want.

**递归解法**：

```c++
class Solution {
public:
    vector<string> letterCombinations(string digits) {
        // store which chars can be used from which index
        
        vector<string> res;
        if(digits.empty()) return res;
        // start with empty string
        string s = "";
        backtrack(digits, alpha, res, s, 0);
        return res;
    }

    void backtrack(string digits, vector< vector<char> > &alpha, vector<string> &res, string s, int idx){
        // 判断递归退出条件
        if(digits.size() == idx){
            res.push_back(s);
            return;
        }
        // 将digits中的字符转为数字
        int x = digits[idx] - '0';
        // 继续递归
        for(int i = 0; i < alpha[x].size(); i++){
            s.push_back(alpha[x][i]);
            backtrack(digits, alpha, res, s, idx+1);
            s.pop_back();
        }
    }
};
```

# 394. Decode Strings
> 题目链接见 https://leetcode.com/problems/decode-string/

Given an encoded string, return its decoded string.

The encoding rule is: `k[encoded_string]`, where the **encoded_string** inside the square brackets is being repeated exactly *k* times. Note that *k* is guaranteed to be a positive integer.

You may assume that the input string is always valid; No extra white spaces, square brackets are well-formed, etc.

Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, *k*. For example, there won't be input like `3a` or `2[4]`.

**Examples:**

```
s = "3[a]2[bc]", return "aaabcbc".
s = "3[a2[c]]", return "accaccacc".
s = "2[abc]3[cd]ef", return "abcabccdcdcdef".
```

**递归解法**：

```c++
class Solution {
public:
    string decodeString(string s) {
        int pos = 0;
        return helper(pos, s);
    }
    
    string helper(int &pos, string s){
        int num = 0;
        string word = "";
        for(; pos < s.size(); pos++){
            char cur = s[pos];
            if(cur == '['){
                string sub_word = helper(++pos, s);
                for(;num>0;num--) word += sub_word;
             // 每进入一个子问题时，word会初始化为空，因此不会带有之前的字符
            }
            else if(isdigit(cur))
                num = num*10 + cur - '0';
            else if(cur == ']')
                return word;
            else
                word += cur;
        }
        return word;
    }
};
```

# 872. Leaf-Similar Trees
> 题目链接见 https://leetcode.com/problems/leaf-similar-trees/

Consider all the leaves of a binary tree.  From left to right order, the values of those leaves form a *leaf value sequence*.

<img src="https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/16/tree.png" width = "300" />

For example, in the given tree above, the leaf value sequence is `(6, 7, 4, 9, 8)`.

Two binary trees are considered leaf-similar if their leaf value sequence is the same.

Return `true` if and only if the two given trees with head nodes `root1` and `root2` are leaf-similar.

**Note**:

- Both of the given trees will have between `1` and `100` nodes.


**递归解法**：

```c++
class Solution {
public:
    bool leafSimilar(TreeNode* root1, TreeNode* root2) {
        string val1, val2;
        helper(root1, val1);
        helper(root2, val2);
        return val1 == val2;
    }
    
    void helper(TreeNode *root, string &vals){
        // return the leaf value sequence from left to right order
        if(!root) return;
        if(!root->left && !root->right){
            vals += to_string(root->val) + '#';
            return;
        }
        if(root->left) helper(root->left, vals);
        if(root->right) helper(root->right, vals);
    } 
};
```

# 979. Distribute Coins in Binary Tree

> 题目链接 https://leetcode.com/problems/distribute-coins-in-binary-tree/

Given the `root` of a binary tree with `N` nodes, each `node` in the tree has `node.val` coins, and there are `N` coins total.

In one move, we may choose two adjacent nodes and move one coin from one node to another.  (The move may be from parent to child, or from child to parent.)

Return the number of moves required to make every node have exactly one coin.

**Example 1**:

<img src="https://assets.leetcode.com/uploads/2019/01/18/tree1.png" width = "200" />

```
Input: [3,0,0]
Output: 2
Explanation: From the root of the tree, we move one coin to its left child, and one coin to its right child.
```

**Example 2**:

<img src="https://assets.leetcode.com/uploads/2019/01/18/tree2.png" width="200" />

```
Input: [0,3,0]
Output: 3
Explanation: From the left child of the root, we move two coins to the root [taking two moves].  Then, we move one coin from the root of the tree to the right child.
```

**思路**：
> https://leetcode.com/problems/distribute-coins-in-binary-tree/solution/

**Intuition**

If the leaf of a tree has 0 coins (an excess of -1 from what it needs), then we should push a coin from its parent onto the leaf. If it has say, 4 coins (an excess of 3), then we should push 3 coins off the leaf. In total, the number of moves from that leaf to or from its parent is `excess = Math.abs(num_coins - 1)`. Afterwards, we never have to consider this leaf again in the rest of our calculation.

**Algorithm**

We can use the above fact to build our answer. Let dfs(node) be the excess number of coins in the subtree at or below this node: namely, the number of coins in the subtree, minus the number of nodes in the subtree. Then, the number of moves we make from this node to and from its children is `abs(dfs(node.left)) + abs(dfs(node.right))`. After, we have an excess of `node.val + dfs(node.left) + dfs(node.right) - 1` coins at this node.



**递归解法**：

```c++
class Solution {
public:
    int res;
    int distributeCoins(TreeNode* root) {
        res = 0;
        dfs(root);
        return res;
    }
    
    int dfs(TreeNode *root){
        // the excess number of coins in the subtree at this node
        if(!root) return 0;
        int L = dfs(root->left), R = dfs(root->right);
        res += abs(L) + abs(R);
        return root->val + L + R - 1;
    }
};
```

# 897. Increasing Order Search Tree

> 题目链接 https://leetcode.com/problems/increasing-order-search-tree/

Given a binary search tree, rearrange the tree in in-order so that the leftmost node in the tree is now the root of the tree, and every node has no left child and only 1 right child.

**Example 1**:
```
Input: [5,3,6,2,4,null,8,1,null,null,null,7,9]

       5
      / \
    3    6
   / \    \
  2   4    8
 /        / \ 
1        7   9

Output: [1,null,2,null,3,null,4,null,5,null,6,null,7,null,8,null,9]


 1
  \
   2
    \
     3
      \
       4
        \
         5
          \
           6
            \
             7
              \
               8
                \
                 9  
```

**Note**:

The number of nodes in the given tree will be between `1` and `100`.
Each node will have a unique integer value from `0` to `1000`.


**递归解法**：

```c++
class Solution {
public:
    TreeNode* increasingBST(TreeNode* root) {
        if(!root) return NULL;
        TreeNode *L = NULL, *R = NULL;
        // 左右节点皆为空，直接返回该节点作为子树的根节点
        if(!root->left && !root->right) return root;
        else{
            if(root->left) L = increasingBST(root->left);
            if(root->right) R = increasingBST(root->right);
            // 将左子树的left赋空，左子树最右侧叶节点的right指向根节点
            if(L){
                L->left = NULL;
                TreeNode *cur = L;
                // 找到左子树最右侧根节点，并将其right指向根节点
                while(cur->right) cur = cur->right;
                cur->right = root;
            }
            // 将根节点的right指向右子树形成的incresingBST，left赋空
            root->left = NULL;
            root->right = R;
        }
        // 如果左子树存在，则返回左子树，否则返回根节点
        return L ? L : root;
    }
};
```

# 109. Convert Sorted List to Binary Search Tree

> 题目链接 https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/

Given a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.

For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.

**Example 1**：
```
Given the sorted linked list: [-10,-3,0,5,9],

One possible answer is: [0,-3,9,-10,null,5], which represents the following height balanced BST:

      0
     / \
   -3   9
   /   /
 -10  5
```

**递归解法**：
```c++
class Solution {
public:
    /*
    基本思路是，先找到单链表的中点，然后递归地将其左边形成一棵BST，右边也形成一棵BST
    */
    TreeNode* sortedListToBST(ListNode* head) {
        if(!head) return NULL;
        if(!head->next) return new TreeNode(head->val);
        // 找到链表的中点
        ListNode *fast = head, *slow = head;
        while(fast && fast->next){
            fast = fast->next->next;
            slow = slow->next;
        }
        ListNode *mid = slow;
        ListNode *last = mid->next;
        // 将中点的两边断开连接
        ListNode *pos = head;
        while(pos->next != mid)
            pos = pos->next;
        pos->next = NULL;
        // 对链表的两部分分别递归
        TreeNode *L = sortedListToBST(head);
        TreeNode *R = sortedListToBST(last);
        TreeNode *root = new TreeNode(mid->val);
        root->left = L; root->right = R;
        return root;
    }
};
```

# 687. Longest Univalue Path

> 题目链接 https://leetcode.com/problems/longest-univalue-path/

Given a binary tree, find the length of the longest path where each node in the path has the same value. This path may or may not pass through the root.

The length of path between two nodes is represented by the number of edges between them.

**Example 1**：
```
Input:

              5
             / \
            4   5
           / \   \
          1   1   5
          
Output: 2
```

**Example 2**:
```
Input:

              1
             / \
            4   5
           / \   \
          4   4   5

Output: 2
```

**递归解法**:

```c++
class Solution {
public:
    int longestUnivaluePath(TreeNode* root) {
        if(!root) return 0;
        int longestPath = 0;
        go(root, longestPath);
        return longestPath;
    }
    // go(root, val)表示从root节点开始，到子树的最长路径
    int go(TreeNode *root, int &length){
        int l = root->left ? go(root->left, length) : 0;
        int r = root->right ? go(root->right, length) : 0;
        l = (root->left && root->left->val == root->val) ? l + 1 : 0;
        r = (root->right && root->right->val == root->val) ? r + 1 : 0;
        length = max(length, l + r);
        return max(l, r);
    }
};
```

# 513. Find Bottom Left Tree Value

> 题目链接 https://leetcode.com/problems/find-bottom-left-tree-value/

Given a binary tree, find the leftmost value in the last row of the tree.

**Example 1**：
```
Input:

    2
   / \
  1   3

Output:
1
```

**Example 2**:
```
Input:

        1
       / \
      2   3
     /   / \
    4   5   6
       /
      7

Output:
7
```
**解题思路**:
既然要找到最下面一层最左边的节点，那使用DFS和BFS都可以。

**解法一： 递归解法(DFS)**:
```c++
class Solution {
public:
    int findBottomLeftValue(TreeNode* root) {
        int maxDepth = 0, leftval = root->val;
        findBottomLeftValue(root, maxDepth, leftval, 0);
        return leftval;
    }
    
    void findBottomLeftValue(TreeNode* root, int &maxDepth, int &leftval, int depth){
        if(!root) return;
        findBottomLeftValue(root->left, maxDepth, leftval, depth + 1);
        findBottomLeftValue(root->right, maxDepth, leftval, depth + 1);
        if(depth > maxDepth){
            maxDepth = depth;
            leftval = root->val;
        }
    }
};
```

**解法二：非递归解法(DFS)**:
```c++
class Solution {
public:
    int findBottomLeftValue(TreeNode* root) {
        int val = root->val;
        stack< pair<TreeNode*, int> > res;
        res.push(pair(root, 1));
        
        int maxDepth = 0;
        while( !res.empty() ){
            pair<TreeNode*, int> node = res.top(); res.pop();
            if(node.first){
                if(node.second > maxDepth){
                    maxDepth = node.second;
                    val = node.first->val;
                }
                // 注意此处先将右节点入栈，再将左节点入栈
                res.push(pair(node.first->right, node.second + 1));
                res.push(pair(node.first->left, node.second + 1));
            }
        }
        return val;
    }
};
```

**解法三：非递归解法(BFS)**:
```c++
class Solution {
public:
    int findBottomLeftValue(TreeNode* root) {
        int leftmost;
        queue<TreeNode*> q;
        q.push(root);
        while(!q.empty()){
            vector<int> level;
            auto size = q.size();
            while(size--){
                TreeNode *node = q.front();
                q.pop();
                if(node->left) q.push(node->left);
                if(node->right) q.push(node->right);
                level.push_back(node->val); 
            }
            leftmost = level[0]; // 记录每一层最左边的值
            level.clear();
        }
        return leftmost;
    }
};
```

# 965. Univalued Binary Tree

> 题目链接 https://leetcode.com/problems/univalued-binary-tree/

A binary tree is univalued if every node in the tree has the same value.

Return `true` if and only if the given tree is univalued.

**Example 1**:

<img src="https://assets.leetcode.com/uploads/2018/12/28/unival_bst_1.png" width="200" />

```
Input: [1,1,1,1,1,null,1]
Output: true
```

**Example 2**:

<img src="https://assets.leetcode.com/uploads/2018/12/28/unival_bst_2.png" width="200" />

```
Input: [2,2,2,5,2]
Output: false
```

**解法一：递归**：

```c++
class Solution {
public:
    bool isUnivalTree(TreeNode* root) {
        if(!root) return true;
        if(root->left && root->val != root->left->val)
            return false;
        if(root->right && root->val != root->right->val)
            return false;            
        return isUnivalTree(root->left) && isUnivalTree(root->right);
    }
};
```

**解法二：迭代(DFS)**：

```c++
class Solution {
public:
    bool isUnivalTree(TreeNode* root) {
        stack<TreeNode *> stk;
        stk.push(root);
        while(!stk.empty()){
            TreeNode *cur = stk.top(); stk.pop();
            if(cur->left){
                if(cur->val != cur->left->val) return false;
                stk.push(cur->left);
            }
            if(cur->right){
                if(cur->val != cur->right->val) return false;
                stk.push(cur->right);
            }
        }
        return true;
    }
};
```

# 450. Delete Node in a BST

> 题目链接 https://leetcode.com/problems/delete-node-in-a-bst/

Given a root node reference of a BST and a key, delete the node with the given key in the BST. Return the root node reference (possibly updated) of the BST.

Basically, the deletion can be divided into two stages:

1. Search for a node to remove.
2. If the node is found, delete the node.

Note: Time complexity should be O(height of tree).

**Example**:
```
root = [5,3,6,2,4,null,7]
key = 3

    5
   / \
  3   6
 / \   \
2   4   7

Given key to delete is 3. So we find the node with value 3 and delete it.

One valid answer is [5,4,6,2,null,null,7], shown in the following BST.

    5
   / \
  4   6
 /     \
2       7

Another valid answer is [5,2,6,null,4,null,7].

    5
   / \
  2   6
   \   \
    4   7
```

**递归解法**:
```c++
class Solution {
public:
    TreeNode* deleteNode(TreeNode* root, int key) {
        if(root == NULL) return NULL;
        
        if(root->val < key){
            root->right = deleteNode(root->right, key);
        }else if(root->val > key){
            root->left = deleteNode(root->left, key);
        }else{
            if(root->left == NULL) return root->right;
            if(root->right == NULL) return root->left;
            // key == root->val
            TreeNode* rightSmallest = root->right;
            while(rightSmallest->left) rightSmallest = rightSmallest->left;
            rightSmallest->left = root->left;
            return root->right;
        }
        return root;
    }
};
```

# 669. Trim a Binary Search Tree

> 题目链接 https://leetcode.com/problems/trim-a-binary-search-tree/

Given a binary search tree and the lowest and highest boundaries as `L` and `R`, trim the tree so that all its elements lies in `[L, R] (R >= L)`. You might need to change the root of the tree, so the result should return the new root of the trimmed binary search tree.

**Example 1**:
```
Input: 
    1
   / \
  0   2

  L = 1
  R = 2

Output: 
    1
      \
       2
```

**Example 2**:
```
Input: 
    3
   / \
  0   4
   \
    2
   /
  1

  L = 1
  R = 3

Output: 
      3
     / 
   2   
  /
 1
```

**递归解法**:
```c++
class Solution {
public:
    TreeNode* trimBST(TreeNode* root, int L, int R) {
        if(root == NULL) return NULL;
        if(root->val < L){
            return trimBST(root->right, L, R);
        }
        else if(root->val > R){
            return trimBST(root->left, L, R);
        }
        else{
            TreeNode* left = trimBST(root->left, L, R);
            TreeNode* right = trimBST(root->right, L, R);
            root->left = left; root->right = right;
            return root;
        }
    }
};
```

# 653. Two Sum IV - Input is a BST

> 题目链接 https://leetcode.com/problems/two-sum-iv-input-is-a-bst/

Given a Binary Search Tree and a target number, return true if there exist two elements in the BST such that their sum is equal to the given target.

**Example 1**:
```
Input: 
    5
   / \
  3   6
 / \   \
2   4   7

Target = 9

Output: True
```

**Example 2**:
```
Input: 
    5
   / \
  3   6
 / \   \
2   4   7

Target = 28

Output: False
```

**解法一（递归）**:
```c++
class Solution {
public:
    set<int> s;
    bool findTarget(TreeNode* root, int k) {
        if(!root) return false;
        if(s.count(k - root->val)) return true;
        s.insert(root->val);
        return findTarget(root->left, k) || findTarget(root->right, k);
    }
};
```

**解法二（DFS + Set）**：
```c++
class Solution {
public:
    bool findTarget(TreeNode* root, int k) {
        if(!root) return false;
        unordered_set<int> s;
        stack<TreeNode*> stk;
        stk.push(root);
        while(stk.size()){
            TreeNode *cur = stk.top();
            stk.pop();
            if(cur){
                if(s.count(k - cur->val))
                    return true;
                else
                    s.insert(cur->val);
                if(cur->right) stk.push(cur->right);
                if(cur->left) stk.push(cur->left);
            }
        }
        return false;
    }
};
```

# 606. Construct String from Binary Tree

> 题目链接 https://leetcode.com/problems/construct-string-from-binary-tree/

You need to construct a string consists of parenthesis and integers from a binary tree with the preorder traversing way.

The null node needs to be represented by empty parenthesis pair "()". And you need to omit all the empty parenthesis pairs that don't affect the one-to-one mapping relationship between the string and the original binary tree.

**Example 1**:
```
Input: Binary tree: [1,2,3,4]
       1
     /   \
    2     3
   /    
  4     

Output: "1(2(4))(3)"

Explanation: Originallay it needs to be "1(2(4)())(3()())", 
but you need to omit all the unnecessary empty parenthesis pairs. 
And it will be "1(2(4))(3)".
```

**Example 2**:
```
Input: Binary tree: [1,2,3,null,4]
       1
     /   \
    2     3
     \  
      4 

Output: "1(2()(4))(3)"

Explanation: Almost the same as the first example, 
except we can't omit the first parenthesis pair to break the one-to-one mapping relationship between the input and the output.
```

**递归解法**：
```c++
class Solution {
public:
    string tree2str(TreeNode* root) {
        if(!root) return "";
        if(!root->left && !root->right) return to_string(root->val);
        if(!root->right) return to_string(root->val) + "(" + tree2str(root->left) + ")";
        return to_string(root->val) + "(" + tree2str(root->left) + ")(" + tree2str(root->right) + ")"; 
    }
};
```

# 538. Convert BST to Greater Tree

> 题目链接 https://leetcode.com/problems/convert-bst-to-greater-tree/

Given a Binary Search Tree (BST), convert it to a Greater Tree such that every key of the original BST is changed to the original key plus sum of all keys greater than the original key in BST.

**Example**:
```
Input: The root of a Binary Search Tree like this:
              5
            /   \
           2     13

Output: The root of a Greater Tree like this:
             18
            /   \
          20     13
```

**递归解法**:
```c++
class Solution {
public:
    int sum = 0;
    TreeNode* convertBST(TreeNode* root) {
        if(root){
            convertBST(root->right);
            root->val += sum;
            sum = root->val;
            convertBST(root->left);
        }
        return root;
    }
};
```

# 671. Second Minimum Node In a Binary Tree

> 题目链接 https://leetcode.com/problems/second-minimum-node-in-a-binary-tree/

Given a non-empty special binary tree consisting of nodes with the non-negative value, where each node in this tree has exactly `two` or `zero` sub-node. If the node has two sub-nodes, then this node's value is the smaller value among its two sub-nodes. More formally, the property `root.val = min(root.left.val, root.right.val)` always holds.

Given such a binary tree, you need to output the **second minimum** value in the set made of all the nodes' value in the whole tree.

If no such second minimum value exists, output -1 instead.

**Example 1**:
```
Input: 
    2
   / \
  2   5
     / \
    5   7

Output: 5
Explanation: The smallest value is 2, the second smallest value is 5.
```

**Example 2**:
```
Input: 
    2
   / \
  2   2

Output: -1
Explanation: The smallest value is 2, but there isn't any second smallest value.
```

**递归解法一**:
```c++
class Solution {
public:
    set<int> S;
    int findSecondMinimumValue(TreeNode* root) {
        secMin(root);
        if(S.size() >= 2){
            return *next(S.begin(), 1);
        }
        return -1;
    }
    
    void secMin(TreeNode* root){
        if(root){
            S.insert(root->val);
            secMin(root->left);
            secMin(root->right);
        }
    }
};
```

**递归解法二**:
```c++
class Solution {
public:
    int findSecondMinimumValue(TreeNode* root) {
        if(!root) return -1;
        int smallval = root->val;
        int res = findSecondMinimumValue(root, smallval);
        return res;
    }
    
    int findSecondMinimumValue(TreeNode* root, int smallval){
        if(root->val != smallval) return root->val;
        int l = root->left ? findSecondMinimumValue(root->left, smallval) : -1;
        int r = root->right ? findSecondMinimumValue(root->right, smallval) : -1;
        if(l == -1) return r;
        if(r == -1) return l;
        return min(l ,r);
    }
};
```

# 572. Subtree of Another Tree

> 题目链接 https://leetcode.com/problems/subtree-of-another-tree/

Given two non-empty binary trees s and t, check whether tree t has exactly the same structure and node values with a subtree of s. A subtree of s is a tree consists of a node in s and all of this node's descendants. The tree s could also be considered as a subtree of itself.

**Example 1**:
```
Given tree s:

     3
    / \
   4   5
  / \
 1   2
Given tree t:
   4 
  / \
 1   2
```
Return **true**, because t has the same structure and node values with a subtree of s.

**Example 2**:
```
Given tree s:

     3
    / \
   4   5
  / \
 1   2
    /
   0
Given tree t:
   4
  / \
 1   2
Return false.
```

**递归解法**:

```c++
class Solution {
public:
    bool isSubtree(TreeNode* s, TreeNode* t) {
        if(!s && !t) return true;
        else if(!s || !t) return false;
        else{
            if(isSame(s, t)) return true;
        return isSubtree(s->left, t) || isSubtree(s->right, t);
        }
    }
    
    bool isSame(TreeNode *s, TreeNode *t){
        if(!s && !t) return true;
        if(!s || !t) return false;
        if(s->val != t->val) return false;
        return isSame(s->left, t->left) && isSame(s->right, t->right);
    }
};
```

# 501. Find Mode in Binary Search Tree

> 题目链接 https://leetcode.com/problems/find-mode-in-binary-search-tree/

Given a binary search tree (BST) with duplicates, find all the mode(s) (the most frequently occurred element) in the given BST.

Assume a BST is defined as follows:

- The left subtree of a node contains only nodes with keys **less than or equal to** the node's key.
- The right subtree of a node contains only nodes with keys **greater than or equal to** the node's key.
- Both the left and right subtrees must also be binary search trees.
 

For example:
Given BST `[1,null,2,2]`,
```
   1
    \
     2
    /
   2
```

return `[2]`.

**Note**: If a tree has more than one mode, you can return them in any order.

**递归解法**:
```c++
class Solution {
private:
    int maxFreq = 0, curFreq = 0, preVal = INT_MIN;
    vector<int> res;
    
public:
    vector<int> findMode(TreeNode* root) {
        inorder(root);
        return res;
    }
    
    void inorder(TreeNode* root){
        if(!root) return;
        inorder(root->left);
        if(preVal == root->val) curFreq++;
        else curFreq = 1;
        if(curFreq > maxFreq){
            res.clear();
            maxFreq = curFreq;
            res.push_back(root->val);
        }
        else if(curFreq == maxFreq){
            res.push_back(root->val);
        }
        preVal = root->val;
        inorder(root->right);
    }
};
```
