# leetcode笔记


## 链表
- **206反转单向链表**
```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        pre, cur = None, head
        while cur:
            cur.next,pre, cur = pre, cur, cur.next
        return pre
```
- **141 环形链表**
```python
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        p1, p2 = head, head
        while(p1 and p2 and p2.next):
            p1, p2 = p1.next, p2.next.next
            if (p1 == p2):
                return True
        return False
```
- **24. 两两交换链表中的节点 （背）**
```python
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if not head or not head.next: return head
        dummy = ListNode(0, head)
        pre = dummy
        while pre.next and pre.next.next:
            a, b = pre.next, pre.next.next
            pre.next, a.next, b.next = b, b.next, a
            pre = a
        return dummy.next 
```
- **234 回文链表**
```python
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        res = []
        cur = head
        while cur:
            res.append(cur.val)
            cur = cur.next
        p1, p2 = 0, len(res)-1
        while p1 <= p2:
            if res[p1] != res[p2]:
                return False
            p1 += 1
            p2 -= 1
        return True
```
方法2是先用快慢指针找出中点，对后半部进行反转，然后判断是否回文，再恢复，能减小空间复杂度
- **138. 复制带随机指针的链表**
```python
#dfs
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        lookup = {}
        def dfs(node):
            if not node: return None
            if node in lookup: return lookup[node]
            copy = Node(node.val)
            lookup[node] = copy
            copy.next, copy.random = dfs(node.next), dfs(node.random)
            return lookup[node]
        return dfs(head)
```
- **148. 排序链表**
  归并排序，找中间节点可以巧用快慢指针的方法做差值
```python
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        if not head or not head.next: return head
        slow, fast = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        node = slow.next
        slow.next = None
        left = self.sortList(head)
        right = self.sortList(node)
        return self.merge(left, right)
    
    def merge(self, l, r):
        dummy = ListNode(0)
        p = dummy
        while l and r:
            if l.val < r.val:
                p.next = l
                p, l = p.next, l.next
            else:
                 p.next = r
                 p, r = p.next, r.next
        # 一边到头了，则剩下的直接合并就好
        p.next = r if r else l
        return dummy.next
```
- **160. 相交链表**
  简单的方法就是用一个list存a的路径，然后遍历b找有没有节点在a里，有的话返回
还有一种方法是双指针的，使用的技巧在于一遍走完了换到另一边，这样ab最后都会走同样的步数，会在相交的地方相遇，如果不相交，则会在链表结尾的None相遇，正常退出循环
```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        p1, p2 = headA, headB
        while p1 != p2:
            p1 = p1.next if p1 else headB
            p2 = p2.next if p2 else headA
        return p1
```
- **328. 奇偶链表**
  双指针递推
```python
class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        if not head or not head.next: return head
        p1, p2, tmp = head, head.next, head.next
        while p1.next.next and p2.next.next:
            p1.next, p2.next = p1.next.next, p2.next.next
            p1, p2 = p1.next, p2.next
        if p1.next.next:
            p1.next = p1.next.next
            p1 = p1.next
        p1.next = tmp
        p2.next = None
        
        return head
```
- **19. 删除链表的倒数第N个节点**
  快慢指针法，但是由于存在删除第一个节点的情况，要建一个dummy node
```python
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode(0, head)
        slow, fast = dummy, dummy
        for _ in range(n):
            fast = fast.next
        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return dummy.next
```

## stack/queue
- **20 判断括号是否有效**
  补充
- **232 225 stack和queue互相实现**
  补充
- **703 返回数据流中的第K大元素**
  1.用优先队列实现
- **239 滑动窗口输出最大值**
  1.用优先队列（大根堆）做，每次删除滑出数字，加入新的数字并维护O（logN），查找最大数字O（1）
  2.数组实现dequeue，数组左边保留最大值

## map映射/set集合
- **哈希冲突的方式**：1.拉链法：重复一个位置放链表
- map是kv对，set只有k（不重复）
- **hashtable vs binary-seatch-tree**:Hash法O（1）但乱序存储，Tree法O（logN）但是有序存储（比如HashMap vs TreeMap   HashSet vs TreeSet）
- **语言实现**
  **Python中**：直接dict（HashMap实现）
  **Java中**：需要指定HashMap或者TreeMap
- **242变位词语**
  1.最简单方法：字符串sorted比较是否相同，快排O（NlogN），N为字符串长度
  2.用Map记字母出现次数，O（N）
- **1 two sum求目标和**
  1.暴力法：O（N^2）
  2.Set：循环x，查找y是否在set集合里（但是得把x自身去掉），O（N）
- **15 three sum     18 four sum求目标和**
  1.暴力法：O（N^3）
  2.Set：循环x，y，查找-(x,y)是否在set集合里（但是得把x,y自身去掉），O（N^2）
  3.sort.find 先整个数组排序，循环x，剩下的左右两边双指针，和大于0：z左移，和小于0:y右移，O（N^2），但比法2省空间（从N到1）

## 树（都得自定义了）
- **二叉搜索树**：左子树比根小，右子树比根大，左右子树都是二叉搜索树
- 普通的二叉搜索树最坏情况会退化到O（N），但红黑树和AVL树不会（始终O（logN））。
- **98 验证二叉排序树**
  1.中序遍历，得到一个升序array，即二叉搜索树（小技巧：不需要每个存，只要判断每次都大于上一次数字就行），O（N）
  2.recursion递归，设计的时候需要额外传两个参数，min，max，判断min小于根，max大于根
- **235 236 最近公共祖先**
  1.路径法：遍历两个节点的路径，取最后出现的公共点即最早公共祖先。（还可以从后往前，取最早出现的祖先，但是需要有父指针），O（N），遍历部分有重复
  2.recursion递归：分别递归查找左右子树里是否含有p和q，具体看代码，O（N），比法1遍历次数少一些
  3.235里用二叉搜索树，只要判断root.val介于p和q之间就是最早公共祖先，法2的递归变种一下就行，非递归也可以写
- **树的遍历**：前/中/后序指的是根节点的访问顺序，在二叉搜索树中会得到不一样的顺序结果（比如中序遍历-升序序列）

## 递归/分治
- 有两个模板，记得多学
- **50. Pow(x, n)**
  1.直接调用库函数，O（1），面试肯定不行
  2.暴力法，循环N次，O（N）
  3.分治：折半用分治递归，O（logN），非递归版使用位运算（这个现在不熟悉，得练）
```python
#分治
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0: return 1
        if n < 0: return 1 / self.myPow(x, -n)
        r = self.myPow(x, int(n/2))
        if n & 1 == 1:
            return r * r * x
        else:
             return r * r
```
```python
#位运算
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0: return 1
        if n < 0: return 1 / self.myPow(x, -n)
        res, tmp =1, x
        while n:
            if n&1:
                res *= tmp
            tmp *= tmp
            n = n >> 1
        return res
```
- **169 找众数**
  1.暴力：双循环，O（N^2）
  2.Map:循环一次dict（x:count_x），求count最大的数，遍历一次即可，O（N）
  3.Sort：先sort排序，遍历一次，找重复次数大于N/2的数字，O（NlogN），排序O（logN）
  4.分治：折半分别找众数，如果两边的众数相同就直接返回任意一个，否则比较两者次数，O（NlogN）

## 贪心greedy
- **122 买卖股票**
  1.DFS：每天都遍历出买和卖的结果，最后选择最优的结果，O（2^N）
  2.贪心：有赚就买卖，只需要遍历一遍，O（N）
  3.DP动态规划：根据每一个天数列一个状态，O（N），但比贪心算法能完成的问题种类更多

## BFS/DFS广度优先搜索
- **BFS广度优先搜索**
  用队列实现，每次访问后加入所有子节点，还要用一个set存放访问标记
- **DFS深度优先搜索**
  用递归实现或者用栈实现，推荐用递归（不用手动维护stack）
- **102 二叉树层序遍历**
  1.BFS：判断当前层级结束的方法：（1）用queue存level信息，但是存储信息太多，不好。（2）Batch Process：预先扫描当前层全部节点。O（N）
  2.DFS：也能实现，但是访问顺序不同，需要记录层数再加，O（N）
- **104 111 求最大/最小深度**
  1.递归
  2.BFS：记录深度level，最大深度即能扫描到的最大深度，最小深度即第一个出现的**叶子节点**的深度，O（N）
  3.DFS：记录深度level，如果是**叶子节点**就更新max和min，O（N）
- **22 有效括号生成**
  1.数学归纳法，很麻烦
  2.递归搜索：每一个位置都2种情况不断分叉，再遍历判断有效的情况，O（2^2N）
  3.剪枝：1.不合法的情况的分支不再递归。2.左右括号的总数不能超过N，超出的情况剪枝。O（2^N）

## 剪枝
- **51 52 N皇后**
  DFS：按层为level，每层level枚举可以放的位置。其中评估当前方案是否可行：（1）暴力，每次全局扫描。（2）剪枝：同一斜线的关系有i+j=C或者i-j=C，同列的话也用一个set来存放，一共要存三个标记位置。复杂度都是阶乘，但是剪枝可以避免大量运算，但N大于20后速度不行。
- **36 37 数独**
  1.朴素DFS：枚举空格就行，dfs（i，j），如果j大于n：j=0，i++，每个空格枚举1-9，check是否合法
  2.剪枝加速：（1）从选项少的格子开始枚举（2）预处理：先用n X n扫描一遍，把每个位置能放的数先扫描好，然后排序（3）从每个格子的可选数里开始枚举
  3.用高级数据结构：DancingLink可以做到

## 二分查找
- 必须要有序、有边界、能用索引访问（不太适合链表）
- **69 求平方根+精确小数位**
  1.二分法查找：要注意x\*x有可能越界，采用m=x/m，或者采用long类型
  2.牛顿迭代法（数学方法）：按公式迭代收敛解

## Trie树-字典树
- 树形结构，哈希树的变种。应用于统计和排序大量字符串，常用于搜索引擎词频统计，可以减少字符串的比较次数，查找效率比哈希表高。
- 用空间换取时间，并且可以利用公共前缀来降低查询时间的开销
- Trie树的边为一个字母，节点为前一个节点加上边的字母，叶子节点是单词，非叶子节点是前缀
- 根节点不包含字符，其他节点只包含一个字符，且每个节点包含的字符都不相同
- **208 Trie树的实现**
  1.没啥技巧，就是多背模板
- **（79）212 单词搜索**
  1.DFS
  2.Trie树：先用候选词建立Trie树，再去枚举board是否有满足Trie树的情况

## 位运算
- 符号 | 描述 |  运算规则  
  -|-|-
  & | 与 | 两个位都为1时才为1 
  \| | 或 | 两个位都为0时才为0 
   ^ | 异或 | 两个位相同时才为1，相反为0 
  ~ | 取反 | 0变1，1变0 
  << | 左移 | 二进制左移N位，高位丢弃，低位补0 
  \>> |  右移| 二进制右移N位，高位补0（有符号位的看编译器，有的补0有的补符号位），低位丢弃 

- **常用的位运算操作**
  X & 1 == 1 OR == 0，判断最后一位是0或1，即判断奇偶性，比模操作更快（X % 2 ==1）
  X = X & (X-1)，清零最低位的1
  X & -X，得到最低位的1
  X ^ X = 0，X ^ 0 = X
- **191 位1的个数**
  1.mod2，如果%2==1，count++，x>>1，O（如数的长度，整数32位）
  2.X = X & (X-1):while（x！=0）:{count++，X = X & (X-1)}，O（位1的个数）
  
- **231 二的次方数**
  1.mod2
  2.开log2看是不是整数
  3.位运算：二的次方数只会存在一个位1，所以判断x!=0 and X & (X-1) == 0
  
- **338 Counting Bits**
  返回0-N里所有每个数的位1的个数
  1.同191
  2.先把所有的数都存在一个数组里，然后循环count[i]=count[ i&(i-1)]，O(N)
  
- **52 N皇后**
  位运算：bits = (~(col|pie|na)) & ((1<<N)-1)，左边是求出可以放的位置，右边是个筛子筛选棋盘的宽度，最后得到可以放的位置。p=bits&(-bits)。递归dfs(n, row+1, col|p, (pie|p)<<1, (na|p)>>1)。bits = bits & (bits-1)
  
## 动态规划DP
- 1.递归+记忆化- >递推
  2.状态的定义: opt[n], dp[n], fib[n]
  3.状态转移方程: opt[n] = best_ of(opt[n-1], opt[n-2], ...
  4.最优子结构
- 和递推相反，自底向上推，还需要有判断+状态存储（为了避免重复计算）
- DP vs回溯vs贪心
  回溯（递归）一重复计算
  贪心一永远局部最优
  DP一记录局部最优子结构/多种记录值
- **70 爬楼梯**
  1.递归： climbStairs(self,int-1)+ climbStairs(self,int-2)，可用一个数组存值，避免重复计算
  2.DP：dp[i] = dp[i - 1] + dp[i - 2]，dp[n]为到第n层的总走法
- **120 Triangle**
  1.递归： Triangle(i,j){
    Triangle(i+1,j)
    Triangle(i+1,j+1)
  },O(2^N)
  2.贪心不可行
  3.DP：
   状态定义：dp[i,j]=点(i,j)到结束的最小距离
   初始方程：最后一行dp[i,j]=val[i,j]
   状态方程：dp[i,j]=min(dp[i+,j],dp[i+1,j+1])+val[i,j]（i从n-2开始倒循环）
   O(M*N)
  **注意：二维方程可以压缩成一维方程——状态压缩**
- **152 乘积最大子序列**
    1.暴力：递归循环
    2.DP：
     状态定义：dp[i]\[2]，dp[i]\[0]=走到第i个元素时，包含i的当前乘积的最大值，dp[i]\[1]为最小值
     初始方程:
     状态方程：dp[i]\[0]=a[i]>=0?dp[i-1]\[0]\*a[i]:dp[i-1]\[1]\*a[i]（即正数=最大值\*自身，负数=最小值\*自身）
                       dp[i]\[1]=a[i]>=0?dp[i-1]\[1]\*a[i]:dp[i-1]\[0]\*a[i]（即正数=最小值\*自身，负数=最小值\*自身）
                       返回dp[i][0]里最大的一个
- **121（1次） 122（无数次） 123（2次） 309（冷静期） 188（k次） 714（含手续费） 股票买卖系列问题**
  1.暴力掠过
  2.dp:状态需要三层\[i]\[k][j]，第一层为cur天数0~n-1，第二层为当前操作的笔数k，第三层为当前是否持有股票0/1（未持有/持有）
   状态定义：dp\[i]\[k]\[j]=第i天时，当前的最大利润
   初始方程：对所有k，dp\[0]\[k]\[0] = 0, dp\[0]\[k][1] = -prices[0]
   状态方程：当前状态和前一天的笔数和是否持有有关
        dp\[i]\[k]\[0]=max(dp\[i-1]\[k]\[0], dp\[i-1]\[k]\[1]+a[i])
        dp\[i]\[k]\[1]=max(dp\[i-1]\[k]\[0]-a[i], dp\[i-1]\[k-1]\[1])
   return：dp\[n-1]\[k][0]
   O(N\*K)
   扩展：（1）冷却情况可以把k改成0，1，记录冷却情况（2）如果可以持有M股，一次交易一股，可以把j改成M，max（买，卖，不动），但是要处理很多边界情况，O(N*K*M)
   **注意：需要对所有k的情况进行一个初始化base case，其次k的循环也是从1开始！**
- **300 最长上升子序列（不用连续）**
  1.DP：两层循环，i循环0到n-1，j循环0到i-1，如果a[i]大于a[j]：dp[i]=dp[j]+1，O（N*N）
  2.二分插入：每一个新的数进来，比右界大就右端插入，比右界小就更新右界，O（NlogN）
- **322 零钱兑换**
  1.暴力法：dfs循环N层遍历（零钱的种类），把所有可能性遍历出来计算最小的count值
  2.DP：类比dp爬楼梯，爬楼梯是1/2步，现在扩展到不同的面值
   状态定义：dp[i]=上到第i时最少的count
   初始方程：最大使用的硬币个数就是所有1元的或者根本无解，无解需要返回-1但是我们比较的是最小值，所以不可以用-1来初始化dp数组，于是使用amount+1来初始化
   状态方程：需要遍历每种硬币的面值，硬币的面值必须要小于当前等于当前的总额才可以放入，并且更新dp[i]为历史状态或者放入一个当前面额中最小的一个
        if coins[j] <= i:
           dp[i] = min(dp[i - coins[j]] + 1, dp[i])
   return：return dp[i] if dp[i] < amount+1 else -1
   O(X\*N)
- **72 编辑距离**单词1变到单词2最小的变动次数
  1.暴力法：对于每一个字符串word1和word2，用dfs或者bfs做单词操作的遍历
  2.DP：字符串问题使用dp的一种解法
   状态定义：dp\[i][j]=i表示word1的前i个字符，j表示word2的前j个字符，整个表示word1的前i个字符替换到word2的前j个字符最少需要的操作次数
   初始方程：
   状态方程：if word1[i-1]==word[j-1] :    **判断字母相同的时候需要-1是因为单词的索引是从0开始的！**
                         dp\[i]\[j]=dp\[i-1]\[j-1]
                     else:三种操作（增删改）的操作次数的最小值
                         dp\[i]\[j]=min(dp\[i-1]\[j], dp\[i]\[j-1], dp\[i-1]\[j-1]) + 1
   return dp\[m]\[n]
   **注意：如果每种操作的开销不同，就在min里面给每个操作加上额外的开销**

## 并查集find&union
- 1.不相交的集合结构
  2.两种优化：（1）增加一个rank表示集合的深度（2）路径压缩：把所有集合都改成两层深度，都指向根节点
- **200 岛屿**求岛屿有几个
  1.染色flood fill：遍历所有的节点，如果节点==1：count++;将节点自身相邻的所有节点（如果相邻节点也是1，也继续染色相邻节点的相邻节点）改成0，count最后就是总的岛屿总数
  2.并查集：
- **547 朋友圈**
  并查集：

## LRU缓存
- 最近使用的排在第一位，置换出最久未使用的
- **146 LRU实现**

## 布隆过滤器 Bloom Filter
- 用一个很长的二进制向量和映射函数来完成
- 可以检查一个元素是否在集合中
- 类似hash表，但是对于存在判断有一定的错误率，但是对不存在的判断是准确的
- 优点是查询时间和空间效率都很高
- 案例：
  1.比特币：redis VS Bloom Filter，redis是存在内存里，而Bloom Filter是来判断是否在不在。
  2.Map Reduce系统

## 面试答题四件套
- 1.Clarification (询问题目细节、边界条件、可能的极端错误情况)
- 2.Possible Solution (所有可能的解法都和面试官沟通一遍)
  Compare Time & Space Complexity (时间复杂度&空间复杂度)
  Optimal Solution (最优解 )
- 3.Coding (写代码)
- 4.Test Cases (测试用例)

## 斐波那契
  - O（N）的方法为多次乘以矩阵[[1,1],[1,0]]


## 摩尔投票法
  - 用一个数存储当前候选人，默认票数为1，若遇到相同的人，票数+1，不同则-1，票数为0的时候换人。
  - 适用于169. 多数元素
