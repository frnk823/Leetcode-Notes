# leetcode笔记


## 链表
- **206 反转单向链表**
```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        pre, cur = None, head
        while cur:
            cur.next,pre, cur = pre, cur, cur.next
        return pre
```
- **92. 反转链表 II**
反转m到n位置的链表，逻辑推理，需要两个标志位来记录两个断点，有点儿绕
```python
class Solution:
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        start = ListNode(0)
        start.next = head
        pre, cur = start, head
        #找到反转起始点并记录
        while m-1 > 0:
            m, n = m-1, n-1
            pre, cur = cur, cur.next
        dummy = pre 
        #反转
        while n > 0:
            cur.next, pre, cur = pre, cur, cur.next
            n -= 1
        #头尾重新接
        dummy.next.next = cur
        dummy.next = pre
        return start.next
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
- **25. K 个一组翻转链表（⚠️注意背）**
比较麻烦，用递归+迭代
```python
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        if not head or not head.next:
            return head
        tail = head
        for i in range(k):
            # 剩余数量小于k的话，则不需要反转
            if not tail:
                return head
            tail = tail.next
        # 反转前k个元素，将返回的头结点记为newHead
        newHead = self.reverse(head, tail)
        # 将head.next 赋为 递归上一步反转得到的newHead
        head.next = self.reverseKGroup(tail, k)
        return newHead
    # 翻转为左闭又开区间，所以本轮操作的尾结点其实就是下一轮操作的头结点
    def reverse(self, head, tail):
        pre = None
        while head != tail:
            tmp = head.next
            head.next = pre
            pre = head
            head = tmp
        return pre
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
- **148. 排序链表（归并排序）**
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
  双指针递推，奇偶先分别连起来，然后保存一下偶数的开头，最后把两条链表串起来，**注意奇数节点可能会多一个**
```python
class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        if not head or not head.next: return head
        p1, p2, tmp = head, head.next, head.next
        while p1.next.next and p2.next.next:
            p1.next, p2.next = p1.next.next, p2.next.next
            p1, p2 = p1.next, p2.next
        #奇数节点可能会多一个
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
- **剑指 Offer 18. 删除链表的节点**
  头节点可能被删，记得dummy开头
```python
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        dummy = ListNode(0)
        dummy.next = head
        pre, cur = dummy, head
        while cur.val != val:
            pre, cur = pre.next, cur.next
        pre.next = cur.next
        return dummy.next
```

## stack/queue
- **20 判断括号是否有效**
```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        dic = {')': '(', '}': '{', ']': '['}
        for i in s:
            if i not in dic:
                stack.append(i)
            else:
                if not stack: return False
                tmp = stack.pop()
                if dic[i] != tmp:
                    return False
        return True if len(stack) == 0 else False
```
- **232 225 stack和queue互相实现**
  补充
- **剑指 Offer 09. 用两个栈实现队列**
```python
class CQueue:

    def __init__(self):
        self.stack1 = []
        self.stack2 = []


    def appendTail(self, value: int) -> None:
        self.stack1.append(value)
        self.stack2 = self.stack1[::-1]


    def deleteHead(self) -> int:
        if not self.stack2:
            return -1
        tmp = self.stack2.pop()
        self.stack1 = self.stack2[::-1]
        return tmp
```
- **703 返回数据流中的第K大元素**
  同样是TopK问题，用优先队列（最小堆）实现
```python
class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.heap = []
        self.k = k
        for val in nums:
            if len(self.heap) < self.k:
                heapq.heappush(self.heap, val)
            elif val > self.heap[0]:
                heapq.heappop(self.heap)
                heapq.heappush(self.heap, val)

    def add(self, val: int) -> int:
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, val)
        elif val > self.heap[0]:
            heapq.heappop(self.heap)
            heapq.heappush(self.heap, val)
        return self.heap[0]
```
- **239 滑动窗口输出最大值**
  1.用优先队列（大根堆）做，每次删除滑出数字，加入新的数字并维护O（logN），查找最大数字O（1）
  2.dequeue实现优先队列，保持数组左边始终是当前最大值
```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        res = []
        queue = collections.deque()
        for i in range(0,len(nums)):
            queue.append(nums[i])
            #长度如果大于k，需要去掉头元素，并调整优先队列
            while len(queue) > k:
                queue.popleft()
                tmp = 0
                while tmp <= len(queue)-1:
                    if queue[tmp] > queue[0]:
                        queue.popleft()
                        tmp = 0
                    tmp += 1
            #长度小于k时也需要调整优先队列
            while queue[0] < nums[i]:
                queue.popleft()
            if i >= k-1:
                res.append(queue[0])
        return res
```
- **215. 数组中的第K个最大元素**
  TopK问题用最小堆实现，堆大小为k，每次加入元素如果**比堆顶大**就替换堆顶元素，调整，最终的堆顶就是第k大的
```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = nums[:k]
        #堆内从尾到头heapify
        for i in range(int((k-1)/2), -1, -1):
            self.heapify(heap, k, i)
        for i in range(k, len(nums)):
            if heap[0] < nums[i]:
                heap[0] = nums[i]
                self.heapify(heap, k, 0)
        return heap[0]
        
    def heapify(self, heap, n, i):
        #n为堆的大小，i为需要调整的节点
        if i >= n: return
        left, right = 2*i+1, 2*i+2
        min_id = i
        if left < n and heap[left] < heap[min_id]:
            min_id = left
        if right < n and heap[right] < heap[min_id]:
            min_id = right
        if min_id != i:
            heap[min_id], heap[i] = heap[i], heap[min_id]
            self.heapify(heap, n, min_id)
```
- **295. 数据流的中位数**
  TopK中位数问题用最大堆+最小堆实现，两个堆的堆顶即是中位的两个数（n为奇数的时候那就是其中一个堆顶元素）
```python
class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        # 初始化大顶堆和小顶堆
        self.max_heap = []
        self.min_heap = []

    def addNum(self, num: int) -> None:
        if len(self.max_heap) == len(self.min_heap):# 先加到大顶堆，再把大堆顶元素加到小顶堆
            heapq.heappush(self.min_heap, -heapq.heappushpop(self.max_heap, -num))
        else:  # 先加到小顶堆，再把小堆顶元素加到大顶堆
            heapq.heappush(self.max_heap, -heapq.heappushpop(self.min_heap, num))

    def findMedian(self) -> float:
        if len(self.min_heap) == len(self.max_heap):
            return (-self.max_heap[0] + self.min_heap[0]) / 2
        else:
            return self.min_heap[0]
```
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
- **128. 最长连续序列**
和字节第一次面试题差不多，但是**不剪枝会超时**，同时set直接手动加入，不要set()转换，否则时间复杂度也会变慢
```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if not nums: return 0
        dic = set()
        for each in nums:
            if each not in dic:
                dic.add(each)
        res = 1    
        for each in dic:
            tmp = 0
            t = each
            if t-1 not in dic:#剪枝！
                while t in dic:
                    if t in dic:
                        tmp += 1
                    t+=1
                res = max(res, tmp)
        return res      
```


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
- **105. 从前序与中序遍历序列构造二叉树**
```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not preorder:
            return
        cur = TreeNode(preorder[0])
        # 当前的根节点
        index = inorder.index(preorder[0])
        # 找到中序遍历里的index，index左边为左子树，右边为右子树
        cur.left = self.buildTree(preorder[1:index + 1], inorder[:index])
        cur.right = self.buildTree(preorder[index + 1:], inorder[index + 1:])
        return cur
```
- **106. 从中序与后序遍历序列构造二叉树**
  类似前一种方法
```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        if not postorder: return
        cur = TreeNode(postorder[-1])
        index = inorder.index(postorder[-1])
        cur.left = self.buildTree(inorder[:index], postorder[:index])
        cur.right = self.buildTree(inorder[index+1:], postorder[index:-1])
        return cur
```
- **889. 根据前序和后序遍历构造二叉树**
  这个比较麻烦，有一些位置需要变
```python
class Solution:
    def constructFromPrePost(self, pre: List[int], post: List[int]) -> TreeNode:
        if not pre: return None
        cur = TreeNode(pre[0])
        if len(pre) == 1: return cur
        index = post.index(pre[1]) + 1
        cur.left = self.constructFromPrePost(pre[1:index+1], post[:index])
        cur.right = self.constructFromPrePost(pre[index+1:], post[index:-1])
        return cur
```
- **144. 二叉树的前序遍历**
  单栈迭代方法，但是需要注意顺序：先右后左
```python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if not root: return []
        res = []
        stack = [root]
        while stack:
            cur = stack.pop()
            res.append(cur.val)
            if cur.right:
                stack.append(cur.right)
            if cur.left:
                stack.append(cur.left)
        return res
```
- **145. 二叉树的后序遍历**
  单栈迭代方法，但是需要注意顺序：先左后右，输出倒序
```python
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        if not root: return []
        stack=[root]
        res=[]
        while stack:
            s=stack.pop()
            res.append(s.val)
            #和前序遍历是相反的
            if s.left:
                stack.append(s.left)
            if s.right:
                stack.append(s.right)
        return res[::-1]
```
- **94. 二叉树的中序遍历**
  单栈迭代方法，模版有些不一样，需要先找到左下角的节点
```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if not root: return []
        res = []
        stack = []
        cur = root
        while cur or stack:
            #一直压栈到左下角节点
            while cur:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            res.append(cur.val)
            cur = cur.right
        return res
```
- **113/剑指 Offer 34. 二叉树中和为某一值的路径**
  dfs
```python
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        if not root: return []
        res = []
        def dfs(node, path, count):
            if not node:
                return
            count += node.val
            if count == sum and not node.left and not node.right:
                res.append(path+[node.val])
                return
            dfs(node.left, path+[node.val], count)
            dfs(node.right, path+[node.val], count)
        dfs(root, [], 0)
        return res
```
- **103. 二叉树的锯齿形层次遍历**
层序遍历+标志正反交错打印level
```python
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        queue = collections.deque()
        res = []
        sym = 0
        queue.append(root)
        while queue:
            level = []
            lenth = len(queue)
            for i in range(lenth):
                cur = queue.popleft()
                if not cur:
                    continue
                level.append(cur.val)
                queue.append(cur.left)
                queue.append(cur.right)
            if level:
                if sym == 0:
                    res.append(level)
                    sym = 1
                elif sym == 1:
                    res.append(level[::-1])
                    sym = 0
        return res
```
- **剑指 Offer 54. 二叉搜索树的第k大节点**
中序遍历BST便有序，但从大到小逆序就可以，递归解决
```python
class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        self.k, self.res = k, 0
        #inorder
        def dfs(root):
            if not root: return
            dfs(root.right)
            self.k -= 1
            if self.k == 0:
                self.res = root.val
                return
            dfs(root.left)
        dfs(root)
        return self.res
```
## 递归/分治
- 有两个模板，记得多学
- **50. Pow(x, n)/剑指 Offer 16. 数值的整数次方**
  1.直接调用库函数，O（1），面试肯定不行
  2.暴力法，循环N次，O（N）
  3.分治：折半用分治递归，O（logN），非递归版使用位运算
```python
#分治法，注意区分奇偶情况
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0: return 1
        if n < 0: return 1 / self.myPow(x, -n)
        r = self.myPow(x, n//2)
        if n & 1 == 1:
            return r * r * x
        else:
             return r * r
```
```python
#位运算    背诵！！！！！！！！！！！！！！！！
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
- **剑指 Offer 12. 矩阵中的路径/79. 单词搜索**
这个写的不好，后续优化一下，改成`nonlocal res`
```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        if not board: return False
        if not word: return True
        dx = [1, -1, 0, 0]
        dy = [0, 0, 1, -1]
        hight = len(board)
        lenth = len(board[0])
        self.res = False
        def dfs(i, j, des):
            if board[i][j] == des[0] and self.res == False:
                if len(des) == 1:
                    self.res = True
                    return
                tmp = board[i][j]
                board[i][j] = '@'
                for k in range(4):
                    x, y = i+dx[k], j+dy[k]
                    if -1<x<hight and -1<y<lenth and board[x][y]!='@':
                        dfs(x, y, des[1:])
                board[i][j] = tmp
        for i in range(hight):
            for j in range(lenth):
                dfs(i, j, word)
        return self.res
```
- **剑指 Offer 13. 机器人的运动范围**
```python
class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:
        dx = [1, 0]
        dy = [0, 1]
        ans = [[0 for _ in range(n)] for _ in range(m)]
        res = 0
        def cal(n):
            s = 0
            while n:
                s += n%10
                n //= 10
            return s

        def dfs(i, j):
            nonlocal res, ans
            ans[i][j] = 1
            res += 1
            for d in range(2):
                x, y = i+dx[d], j+dy[d]
                if -1<x<m and -1<y<n and cal(x)+cal(y)<=k and ans[x][y] == 0:
                    dfs(x, y)
        dfs(0, 0)
        return res
```


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
  1.二分法查找：要注意`x*x`有可能越界，采用m=x/m，或者采用long类型
  2.牛顿迭代法（数学方法）：按公式迭代收敛解
- **153. 寻找旋转排序数组中的最小值**
  这一题看起来直接遍历就可以解决，但是一般题不可能这么简单，于是需要想办法进行优化
我们可以看出这个数组自身是有序的，于是是可以用二分法进行优化的，思路类似二分查找，但是在处理逻辑上需要对题目进行优化，可以看出：如果最小值在区间段内，那么最左是大于最右的
```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        low, high = 0, len(nums)-1
        while low < high:
            mid = (low + high) >> 1
            if nums[mid] > nums[high]:
                low = mid + 1
            else:
                high = mid 
        return nums[low]
```
- **剑指 Offer 11. 旋转数组的最小数字/154. 寻找旋转排序数组中的最小值 II（⚠️重点注意else！）**
注意！！！因为数字可以重复，所以遇到相等的情况不能直接变界，执行high- - 
```python
class Solution:
    def minArray(self, numbers: List[int]) -> int:
        low, high = 0, len(numbers)-l
        while low < high:
            mid = (low + high) >> 1
            if numbers[mid] > numbers[high]:
                low = mid + 1
            elif numbers[mid] < numbers[high]:
                high = mid 
            #特别注意这个情况
            else: high -= 1
        return numbers[low]
```
- **34. 在排序数组中查找元素的第一个和最后一个位置**
二分查找，找边界的时候==情况不返回，而是继续缩小边界，注意右边界查找的时候有一个+1
```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if not nums: return [-1, -1]
        # 寻找左边界
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + right >> 1
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        if nums[left] != target:
            return [-1, -1]
        res = [left]

        # 寻找右边界，注意此时的 left 是左边界的索引值
        right = len(nums) - 1
        while left < right:
            mid = left + right + 1 >> 1
            if nums[mid] > target:
                right = mid - 1
            else:
                left = mid
        res.append(right)
        return res
```

## Trie树-字典树
- 树形结构，哈希树的变种。应用于统计和排序大量字符串，常用于搜索引擎词频统计，可以减少字符串的比较次数，查找效率比哈希表高。
- 用空间换取时间，并且可以利用公共前缀来降低查询时间的开销
- Trie树的边为一个字母，节点为前一个节点加上边的字母，叶子节点是单词，非叶子节点是前缀
- 根节点不包含字符，其他节点只包含一个字符，且每个节点包含的字符都不相同
- **208 Trie树的实现**
  1.没啥技巧，就是多背模板
- **（79升级版）212 单词搜索**
  1.DFS
  2.Trie树：先用候选词建立Trie树，再去枚举board是否有满足Trie树的情况
```python
class Solution:
    dx = [1, -1, 0, 0]
    dy = [0, 0, 1, -1]

    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        if not words or not board or board==[] : return []
        self.res = set()
        #Trie tree
        self.root = {}
        self.is_end = '#'
        #将board存进Trie树中
        for word in words:
            cur = self.root
            for each in word:
                if each not in cur:
                    cur[each] = {}
                cur = cur[each]
            cur[self.is_end] = self.is_end    
        self.width = len(board)
        self.lenth = len(board[0])
        #开始递归
        for i in range(self.width):
            for j in range(self.lenth):
                if board[i][j] in self.root:
                    self.dfs(board, i, j, '', self.root)
        
        return list(self.res)
    #dfs：使用过的位置需要置空，递归完需要还原
    def dfs(self, board, i, j, cur_word, cur_node):
        cur_word += board[i][j]
        cur_node = cur_node[board[i][j]]

        if self.is_end in cur_node:
            self.res.add(cur_word)
        
        temp = board[i][j]
        board[i][j] = '@'

        for k in range(4):
            x, y=i+self.dx[k], j+self.dy[k]
            if -1<x<self.width and -1<y<self.lenth and board[x][y]!='@' and board[x][y] in cur_node:
                    self.dfs(board, x, y, cur_word, cur_node)
        board[i][j] = temp
```

## 位运算
- 符号 | 描述 |  运算规则  
  -|-|-
  `&` | 与 | 两个位都为1时才为1 
  `|` | 或 | 两个位都为0时才为0 
  `^` | 异或 | 两个位相同时才为1，相反为0 
  `~` | 取反 | 0变1，1变0 
  `<<` | 左移 | 二进制左移N位，高位丢弃，低位补0 
  `>>` |  右移| 二进制右移N位，高位补0（有符号位的看编译器，有的补0有的补符号位），低位丢弃 

- **常用的位运算操作**
 ` X & 1 == 1 OR == 0`，判断最后一位是0或1，即判断奇偶性，比模操作更快（X % 2 ==1）
  `X = X & (X-1)`，清零最低位的1
  `X & -X`，得到最低位的1
  `X ^ X = 0`，`X ^ 0 = X`

- **191 位1的个数/剑指 Offer 15. 二进制中1的个数**
  1.mod2，`if %2==1:count++，x>>1`，O（如数的长度，整数32位）
  2.利用`X & (X-1)`消除最低位的1，`O（位1的个数）`
```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        res = 0
        while n:
            n = n & (n-1)
            res += 1
        return res
```

- **231 二的次方数**
  1.mod2
  2.开log2看是不是整数
  3.位运算：二的次方数只会存在一个位1，所以判断`x!=0 and X & (X-1) == 0`
  
- **338 Counting Bits**
  返回0-N里所有每个数的位1的个数
  1.同191
  2.先把所有的数都存在一个数组里，然后循环`count[i]=count[ i&(i-1)]`，`O(N)`
  
- **52 N皇后**
  位运算：bits = (~(col|pie|na)) & ((1<<N)-1)，左边是求出可以放的位置，右边是个筛子筛选棋盘的宽度，最后得到可以放的位置。p=bits&(-bits)。递归dfs(n, row+1, col|p, (pie|p)<<1, (na|p)>>1)。bits = bits & (bits-1)
  
## 动态规划DP
- 1.递归+记忆化- >递推
  2.状态的定义: `opt[n]`, `dp[n]`, `fib[n]`
  3.状态转移方程: `opt[n] = best_ of(opt[n-1], opt[n-2], ...`
  4.最优子结构
- 和递推相反，自底向上推，还需要有判断+状态存储（为了避免重复计算）
- DP vs 回溯 vs 贪心
  回溯（递归）一重复计算
  贪心一永远局部最优
  DP一记录局部最优子结构/多种记录值
- **70 爬楼梯**
  1.递归： climbStairs(self,int-1)+ climbStairs(self,int-2)，可用一个数组存值，避免重复计算
  2.DP：`dp[i] = dp[i - 1] + dp[i - 2]`，`dp[n]`为到第n层的总走法
- **120. 三角形最小路径和**
  1.递归：` Triangle(i,j){
    Triangle(i+1,j)
    Triangle(i+1,j+1)
  }`,`O(2^N)`
  2.贪心不可行
  3.DP：
   dp方法需要自底向上推导，可以将二维方程压缩成一维
   状态定义：`dp[i,j]=点(i,j)到结束的最小距离`
   初始方程：最后一行`dp[i,j]=val[i,j]`
   状态方程：`dp[i,j]=min(dp[i+1,j],dp[i+1,j+1])+val[i,j]`（i从n-2开始倒循环）
   O(M*N)
  **注意：二维方程可以压缩成一维方程——状态压缩**
```python
class Solution:
    #没有压缩的二维dp
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        if not triangle: return 0
        high = len(triangle)
        dp = triangle.copy()
        for i in range(high-2, -1 , -1):
            for j in range(len(triangle[i])):
                dp[i][j] = triangle[i][j] + min(dp[i+1][j],dp[i+1][j+1])
        return dp[0][0]
    # 压缩后的一维dp
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        if not triangle: return 0
        high = len(triangle)
        dp = triangle[-1]
        for i in range(high-2, -1 , -1):
            for j in range(len(triangle[i])):
                dp[j] = triangle[i][j] + min(dp[j],dp[j+1])
        return dp[0]
```
- **骨牌填充问题**
1*n的格子放1*1、1*2、1*3的骨牌
2*n的格子放1*2的骨牌
都可以类似爬楼梯递推，第一个骨牌的摆放位置决定了剩下的格子能放的位置，即类似斐波那契数列的递推法
- **53. 最大子序和**
```python #dp
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        dp = nums.copy()
        dp[0] = nums[0]
        for i in range(1, len(nums)):
            dp[i] = max(nums[i], dp[i-1]+nums[i])
        return max(dp)
```
```python #正数增益（推荐）
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        res, sum = nums[0], 0
        for num in nums:
            if sum > 0:
                sum += num
            else:
                sum = num
            res = max(res, sum)
        return res
```
- **152 乘积最大子序列**
  1.暴力：递归循环
  2.DP：
   状态定义：`dp[i][2]`，`dp[i][0]`=走到第i个元素时，包含i的当前乘积的最大值，`dp[i][1]`为最小值
   初始方程:
   状态方程：`dp[i][0]=a[i]>=0?dp[i-1][0]*a[i]:dp[i-1][1]*a[i]（即正数=最大值*自身，负数=最小值*自身）
             dp[i][1]=a[i]>=0?dp[i-1][1]*a[i]:dp[i-1][0]*a[i]（即正数=最小值*自身，负数=最小值*自身）
             return dp[i][0]里最大的一个`
- **121（1次） 122（无数次） 123（2次） 309（冷静期） 188（k次） 714（含手续费） 股票买卖系列问题**
  1.暴力掠过
  2.dp:状态需要三层`dp[i][k][j]`，第一层为cur天数0~n-1，第二层为当前操作的笔数k，第三层为当前是否持有股票0/1（未持有/持有）
   状态定义：`dp[i][k][j]=第i天时，当前的最大利润`
   初始方程：对所有k，`dp[0][k][0] = 0, dp[0][k][1] = -prices[0]`
   状态方程：当前状态和前一天的笔数和是否持有有关
        `dp[i][k][0]=max(dp[i-1][k][0], dp[i-1][k][1]+a[i])
      dp[i][k][1]=max(dp[i-1][k][0]-a[i], dp[i-1][k-1][1])
      return：dp[n-1][k][0]`
   `O(N*K)`
   扩展：（1）冷却情况可以把k改成0，1，记录冷却情况（2）如果可以持有M股，一次交易一股，可以把j改成M，max（买，卖，不动），但是要处理很多边界情况，O(N*K*M)
   **注意：需要对所有k的情况进行一个初始化base case，其次k的循环也是从1开始！**
- **300 最长上升子序列（不用连续）**
  1.DP：两层循环，i循环0到n-1，j循环0到i-1，如果a[i]大于a[j]：dp[i]=dp[j]+1，O（N*N）
  2.二分插入：每一个新的数进来，比右界大就右端插入，比右界小就更新右界，O（NlogN）
- **322 零钱兑换**
  1.暴力法：dfs循环N层遍历（零钱的种类），把所有可能性遍历出来计算最小的count值
  2.DP：类比dp爬楼梯，爬楼梯是1/2步，现在扩展到不同的面值
   状态定义：`dp[i]`=上到第i时最少的count
   初始方程：最大使用的硬币个数就是所有1元的或者根本无解，无解需要返回-1但是我们比较的是最小值，所以不可以用-1来初始化dp数组，于是使用amount+1来初始化
   状态方程：需要遍历每种硬币的面值，硬币的面值必须要小于当前等于当前的总额才可以放入，并且更新dp[i]为历史状态或者放入一个当前面额中最小的一个
        if coins[j] <= i:
           dp[i] = min(dp[i - coins[j]] + 1, dp[i])
   return：return dp[i] if dp[i] < amount+1 else -1
   `O(X*N)`
- **72 编辑距离**单词1变到单词2最小的变动次数
  1.暴力法：对于每一个字符串word1和word2，用dfs或者bfs做单词操作的遍历
  2.DP：字符串问题使用dp的一种解法
   状态定义：`dp[i][j]`,i表示word1的前i个字符，j表示word2的前j个字符，整个表示word1的前i个字符替换到word2的前j个字符最少需要的操作次数
   初始方程：
   状态方程：`if word1[i-1]==word[j-1] :    **判断字母相同的时候需要-1是因为单词的索引是从0开始的！**
                 dp[i][j]=dp[i-1][j-1]
             else:三种操作（增删改）的操作次数的最小值
                 dp[i][j]=min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
             return dp[m][n]`
   **注意：如果每种操作的开销不同，就在min里面给每个操作加上额外的开销**
- **5. 最长回文子串**
  `dp[i][j]`状态表示i到j是否是回文串，只有当回文字串加上前后相同字母的时候才是回文字串，但是状态转移的时候不能直接i，j循环，那样会丢失部分状态，而是按照字串长度l循环，j = i + l，先循环l后循环i，l为0和1的时候单独处理。
```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s or len(s) == 1: return s
        length = len(s)
        res = ''
        dp = [[0 for _ in range(length)] for _ in range(length)]
        for l in range(length):
            for i in range(length):
                j = i + l
                if j > length-1:
                    break
                if l == 0: 
                    dp[i][j] = 1
                elif l == 1 and s[i] == s[j]:
                    dp[i][j] = 1
                elif dp[i + 1][j - 1] and s[i] == s[j]:
                    dp[i][j] = 1
                if dp[i][j] and l+1 > len(res):
                    res = s[i:j+1]
        return res
```
- **139. 单词拆分**
```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        lenth = len(s)
        dp = [False]*(lenth+1)
        dp[0] = True
        for i in range(lenth):
            for j in range(i+1, lenth+1):
                #因为区间是右开，所以结束位得+1才能取到低
                if(dp[i] and (s[i:j] in wordDict)):
                    dp[j] = True
        return dp[-1]
```
## 并查集find&union
- 1.不相交的集合结构
  2.两种优化：（1）增加一个rank表示集合的深度（2）路径压缩：把所有集合都改成两层深度，都指向根节点
- **200 岛屿 ** 求岛屿有几个
  1.染色flood fill：遍历所有的节点，如果节点==1：count++;将节点自身相邻的所有节点（如果相邻节点也是1，也继续染色相邻节点的相邻节点）改成0，count最后就是总的岛屿总数
  2.并查集：
- **547 朋友圈**
  并查集
```python
class Solution:
    def findCircleNum(self, M: List[List[int]]) -> int:
        if not M: return 0
        roots = [i for i in range(len(M))]
        self.count = len(M)

        def find(a):
            if roots[a] != a:
                roots[a] = find(roots[roots[a]])
            return roots[a]
        
        def union(a, b):
            roots[find(b)] = find(a)
            self.count -= 1

        for i in range(len(M)):
            for j in range(i+1, len(M)):
                if M[i][j] and find(i)!=find(j):
                    union(i, j)
        return self.count  
```

- **64. 最小路径和**
```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dp = [[0]*n for _ in range(m)]
        dp[0][0] = grid[0][0]
        for i in range(1,m):
            dp[i][0] = dp[i-1][0] + grid[i][0]
        for j in range(1,n):
            dp[0][j] = dp[0][j-1] + grid[0][j]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        return dp[-1][-1]
```

## LRU缓存
- 最近使用的排在第一位，置换出最久未使用的
- **146 LRU实现**
```python
class LRUCache:

    def __init__(self, capacity: int):
        self.lru = {}
        self.queue = []
        self.lenth = capacity

    def get(self, key: int) -> int:
        if key in self.lru:
            self.queue.remove(key)
            self.queue.append(key)
            return self.lru[key]
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        self.lru[key] = value
        if key in self.queue:
            self.queue.remove(key) 
        self.queue.append(key)
        if len(self.lru) > self.lenth:
            self.lru.pop(self.queue[0])
            self.queue.remove(self.queue[0])
```

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

## 字符串处理
- **剑指 Offer 17.打印从1到最大的n位数**
法1:最简单的就是用pow进行递增打印，pow函数还可以用分治法进行快速实现
```python
class Solution:
    def printNumbers(self, n: int) -> List[int]:
        res = []
        for i in range(1, self.myPow(10,n)):
            res.append(i)
        return res
    #快速pow要背下来！！！
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
法2:遇到大数越界的问题（python其实还不会），需要转换成字符串来处理
```python
def printNumbers(self, n: int) :
        res=[]
        temp=['0']*n
        def helper(index):
            if index==n:
                res.append(int(''.join(temp)))
                return
            for i in range(10):
                temp[index]=chr(ord("0")+i)
                helper(index+1)
        helper(0)
        return res[1:]
```

- **剑指 Offer 21. 调整数组顺序使奇数位于偶数前面**
类似快排的方法来一个一个交换
```python
class Solution:
    def exchange(self, nums: List[int]) -> List[int]:
        l, r = 0, len(nums)-1
        while l < r:
            while l < r and nums[r]&1 == 0:
                r -= 1
            while l < r and nums[l]&1 == 1:
                l += 1
            nums[l], nums[r] = nums[r], nums[l]
        return nums
```

- **41. 缺失的第一个正数**
原地置换，将数字置换回正确的位置上，再遍历找出缺失的数字，时间复杂度`O(N)`，空间复杂度`O(1)`
```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        if not nums: return 1
        n = len(nums)
        for i in range(n):
            #第二个判断是为了避免死循环
            while 0 < nums[i] < n and nums[nums[i]-1] != nums[i]:
                nums[i], nums[nums[i]-1] = nums[nums[i]-1], nums[i]
        for i in range(n):
            if nums[i] != i+1:
                return i+1
        return n+1
```


## 斐波那契
  - O（N）的方法为多次乘以矩阵[[1,1],[1,0]]


## 摩尔投票法
  - 用一个数存储当前候选人，默认票数为1，若遇到相同的人，票数+1，不同则-1，票数为0的时候换人。
  - 适用于169. 多数元素
```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        if not nums: return
        candidate, votes = 0, 0
        for i in nums:
            if votes == 0:
                candidate = i
            votes += 1 if i == candidate else -1
        return candidate
```

## 马拉车法

## 约瑟夫环
- **剑指62  圆圈中最后剩下的数字**
循环递推，背好递推公式
```python
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        f = 0
        for i in range(2, n+1):
            f = (f + m) % i
        return f
```

## 中文数字转换阿拉伯数字
少考虑了“三千九”这种特殊写法，可以用sym标志位改，但是会和含“零”情况有点冲突，有空再思考
```python
def trans(s):                               
    if s[0]=='十':                           
        s = '一' + s                         
    num = 0                                 
    temp = 0                                
    sym = 0   #存放当前最大的权值，如果不是按照从大到小变化就特殊处理                              
    for i in s:                             
        if i in number:                     
            temp = number[i]                
        elif i in danwei:                   
            if sym == 0 or weight[i] < sym: 
                temp *= danwei[i]           
                num += temp                 
                temp = 0                    
                sym = weight[i]             
            else:                           
                num *= danwei[i]            
                sym = 0                     
        elif i == '零':                      
            pass                            
    if temp != 0:                           
        # num += temp * danwei[sym-1]       
        num += temp                         
    return num                              
```

## 正则匹配
- **剑指 Offer 19. 正则表达式匹配**
⚠️前面需要加一个`#`，原因未知，盲猜是因为用到了`dp[i-1][j-1]`，但是应该可以改成不用`#`的。
```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        s, p = '#'+s, '#'+p
        m, n = len(s), len(p)
        dp = [[False]*n for _ in range(m)]
        dp[0][0] = True
        
        for i in range(m):
            for j in range(1, n):
                #s为空的时候，只能匹配'X*'的情况，需要两个and
                if i == 0:
                    dp[i][j] = j > 1 and p[j] == '*' and dp[i][j-2]
                #当前位相同或者当前位是'.'（匹配任何非/n字符）
                elif p[j] in [s[i], '.']:
                    dp[i][j] = dp[i-1][j-1]
                #当前位为'*'时有两种情况，'X*'或者'.*'
                elif p[j] == '*':
                    dp[i][j] = j > 1 and dp[i][j-2] or p[j-1] in [s[i], '.'] and dp[i-1][j]
                else:
                    dp[i][j] = False
        return dp[-1][-1]
```

- **剑指 Offer 20. 表示数值的字符串**
DFA/regex来解
```python
import re
class Solution:
    def isNumber(self, s: str) -> bool:
        res = re.match(r'^[+-]?(\d+\.?\d*|\.\d+)([Ee][+-]?\d+)?$',s.strip())
        return True if res else False
```

## 数学推导
- **剑指 Offer 14- I. 剪绳子/343. 整数拆分**
1.数学归纳法推导出等分成3的倍数，乘积最大
2.还可以用贪心法解，将n为2-6的情况枚举，超过这个长度的按照贪心切分后进行计算（常规计算）
```python
class Solution:
    def cuttingRope(self, n: int) -> int:
        res = 0
        if n == 2: return 1
        if n == 3: return 2
        a, b = n//3, n%3
        if b == 0:
            return int(math.pow(3, a))
        if b == 1:
            return int(math.pow(3, a-1)*4)
        if b == 2:
            return int(math.pow(3, a)*2)
```
- **剑指 Offer 14- II. 剪绳子 II**
按照旧情况会出现长度超标现象，需要不断循环取余，pow操作也可以转换为分治法来加速
```python
class Solution:
    def cuttingRope(self, n: int) -> int:
        res = 0
        if n == 2: return 1
        if n == 3: return 2
        def cycle(x, a):
            s = 1
            while a:
                s *= x
                s %= 1000000007
                a -= 1 
            return s%1000000007

        a, b = n//3, n%3
        if b == 0:
            return cycle(3, a)
        if b == 1:
            return cycle(3, a-1)*4%1000000007
        if b == 2:
            return cycle(3, a)*2%1000000007
```

**470. 用 Rand7() 实现 Rand10()**
牢记`(rand7()-1)*7+rand7()`这个规律

```python
class Solution:
    def rand10(self):
        """
        :rtype: int
        """
        res = (rand7()-1)*7+rand7()
        while res > 10:
            res = (rand7()-1)*7+rand7()
        return res
```

还可以充分利用有效区间进行取模优化

```python
class Solution:
    def rand10(self):
        """
        :rtype: int
        """
        res = (rand7()-1)*7+rand7()
        while res > 40:
            res = (rand7()-1)*7+rand7()
        return 1+res%10
```



## 其他

- **14. 最长公共前缀**
法1:python特性，元组拆包实现，比较骚，但是要注意一旦出现非前缀要记得break，否则会把后缀都加到结果里
```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs: return ''
        res = ''
        for i in zip(*strs):
            tmp = set(i)
            if len(tmp) == 1:
                res += i[0]
            else:
                break
        return res
```
法2:用最大和最小字符串来比较，简单粗暴，复杂度最低
```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs: return ""
        str0 = min(strs)
        str1 = max(strs)
        for i in range(len(str0)):
            if str0[i] != str1[i]:
                return str0[:i]
        return str0
```



## 字节手撕整理
https://www.nowcoder.com/discuss/455003?type=post&order=create&pos=&page=1&channel=1011&source_id=search_post 