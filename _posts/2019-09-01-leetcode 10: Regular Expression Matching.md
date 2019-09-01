---
layout:     post
title:      leetcode 10 - Regular Expression Matching
subtitle:   动态规划学习笔记
date:       2019-09-01
author:     yangwf
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - leetcode
    - DP

---
# Leetcode 10. Regular Expression Matching

#### Description

Given an input string (`s`) and a pattern (`p`), implement regular expression matching with support for `'.'` and `'*'`.

```
'.' Matches any single character.
'*' Matches zero or more of the preceding element.
```

The matching should cover the **entire** input string (not partial).

**Note:**

- `s` could be empty and contains only lowercase letters `a-z`.
- `p` could be empty and contains only lowercase letters `a-z`, and characters like `.` or `*`.

#### Solution

- 特殊情况
  - p为空时，s也应为空
  - s为空时，p可不为空

- 重点分析

  本题的难点在于对`'*'`的处理，其可匹配0次，也可匹配多次。对于0次情况，直接跳过即可；对于多次情况，首先匹配一次，然后继续用`'*'`对文本进行匹配，递归进行这两次选择，达到目的。

- 递归求解

```python
class Solution:
  def isMatch(self, s: str, p: str) -> bool:
    if not p:
      return not s
    first = bool(s) and p[0] in {'.', s[0]}
    if len(p) >= 2 and p[1] == '*':
      return self.isMatch(s, p[2:]) or first and self.isMatch(s[1:], p)
    else:
      return first and self.isMatch(s[1:] ,p[1:])
```

- DP求解

```python
class Solution:
  def isMatch(self, s: str, p: str) -> bool:    
    m = len(s)
    n = len(p)
    dp = [[False]*(n+1) for _ in range(m+1)]
    dp[m][n] = True
    # i从m开始用来处理text的边界问题
    for i in range(m,-1,-1):
      for j in range(n-1,-1,-1):
        first = i < len(s) and p[j] in {s[i], '.'}
        if j+1 < n and p[j+1] == '*':
          dp[i][j] = dp[i][j+2] or first and dp[i+1][j]
        else:
          dp[i][j] = first and dp[i+1][j+1]            
    return dp[0][0]
```

#### Summary

- 处理`'*'`的方式
- DP公式在两个字符串匹配问题的应用
- 边界问题DP数组➕1
- txet边界问题的处理
