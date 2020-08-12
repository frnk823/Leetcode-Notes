# Java面试题

## Java内存空间
Java的内存需要划分成为5个部分:
- **栈(Stack)** :存放的都是方法中的局部变量。**方法的运行一定要在栈当中运行。**
      局部变量:方法的参数，或者是方法{}内部的变量
      作用域: 一旦超出作用域，立刻从栈内存当中消失。
- **堆(Heap)** :**凡是new出来的东西，都在堆当中。**
      堆内存里面的东西都有一个地址值: 16进制
      堆内存里面的数据，都有默认值。规则:
          如果是整数     默认为0
          如果是浮点数   默认为0.0
          如果是字符     默认为'\u0000'
          如果是布尔     默认为false
          如果是引用类型  默认为null
- **方法区(Method Area) **:存储.class相关信息，包含方法的信息。
- 本地方法栈(Native Method Stack) :与操作系统相关。
- 寄存器(pc Register) :与CPU相关。

## 字符串
-**字符串常量池**
只有直接双引号生成的字符串才在字符串池中，new的不在池子里
字符串常量池在**堆**中 

- *boolean String.equals(Object obj)*：比较两个字符串的内容是否**相同**
**(注：如果和常量字符串比较建议放在外面，不要放在括号里)** 
- *boolean String.equalsIgnoreCase(String str)*：比较两个字符串的内容是否相同**且忽略大小写**
- *String String.concat(String str)*：拼接两个字符串
- *int String.charAt(int index)*：获取指定索引处的字符
- *int String.indexOf(String str))*：查找参数字符串在本字符串中首次出现的索引位置，没有则返回-1
- *String String.substring(int index)*：截取从参数位置开始到结束的字符串
- *String String.substring(int begin,int end)*：截取从参数位置的字符串，左闭右开区间
- *char[ ] String.toCharArray(String)*：将当前字符串拆分成char数组
- *byte[ ] String.getBytes(String )*：将当前字符串拆分成byte数组
- *String String.replace(CharSequence string1,CharSequence string1)*：替换所有出现的字符串 
**(注：CharSequence是接口，可以接收字符串)**
- *String[ ] String.split(String regex)*：按照规则切分字符串**(注：regex是正则表达式)**

## static
- 如果一个成员变量使用了static关键字，那么这个变量不再属于对象自己，而是属于所在的类。**多个对象共享同一份数据。**
- 如果一个成员方法使用了static关键字，那么这个变量不再属于对象自己，而是属于所在的类。**对于静态方法来说，既可以通过对象名来调用方法，也可以通过类名来调用方法（建议用类名调用）。**
- 静态不可以访问非静态（**因为在内存中先有静态内容，后有非静态内容**）
- 静态里不可使用this关键字
- 静态代码块（在类中）：static｛｝静态代码块在第一次构造该类的时候只执行一次且有且执行一次。**典型用途：用来一次性地对成员变量进行赋值（尤其在使用JDBC时）**

## Arrays
- *String Arrays.toString(Arrays)*：将参数数组转换成字符串（默认格式[元素1,元素2,元素3....]）
- *void Arrays.sort(Arrays)*：将数组按照升序排序

  （**注：如果是数值，按照数值大小升序排序；如果是字符串，按照字母升序排序；如果是自定义类型，需要有Comparable或者Comparator接口的支持**）

## Math
- *double Math.abs(double num)*：绝对值
- *double Math.ceil(double num)*：向上取整
- *double Math.floor(double num)*：向下取整
- *long Math.round(double num)*：四舍五入

## HashMap
https://github.com/AobingJava/JavaFamily/blob/master/docs/basics/HashMap.md
- Key-Value在Java7叫Entry，在Java8中叫Node

- **Java8之前用头插法，Java8开始用尾插法**

- **resize**
  Capacity：HashMap当前长度。
  LoadFactor：负载因子，默认值0.75f。
  扩容分为两步：
   1.扩容：创建一个新的Entry空数组，长度是原数组的2倍。
   2.ReHash：遍历原Entry数组，把所有的Entry重新Hash到新数组。
  
- **为什么要重新Hash？**
  因为长度扩大以后，Hash的规则也随之改变。
  
- **为什么之前用头插法，java8之后改成尾插？**
  单链表的头插入方式，同一位置上新元素总会被放在链表的头部位置，**多个线程**插入resize后可能会出现环形链表，从而引发Infinite Loop。**使用头插**会改变链表的上的顺序，但是如果**使用尾插**，在扩容时会保持链表元素原本的顺序，就不会出现链表成环的问题了。
  Java7在多线程操作HashMap时可能引起**死循环**，原因是扩容转移后前后链表顺序倒置，在转移过程中修改了原来链表中节点的引用关系。
  Java8在同样的前提下并不会引起死循环，原因是扩容转移后前后链表顺序不变，保持之前节点的引用关系。**但这并不意味着HashMap线程安全，是非线程安全的**
  
- **HashMap的默认初始化长度是多少？为什么？**
  16，源码里用的是1<<4（位运算比乘方运算快）
  index的计算公式：index = HashCode（Key） & （Length- 1），当Length是2的幂的时候，**Length-1的值是所有二进制位全为1**，这种情况下，index的结果等同于**HashCode后几位的值**。只要输入的HashCode本身分布均匀，Hash算法的结果就是均匀的。
  
- **HashMap是否是线程安全的？如果不是怎么办？**
  线程不安全。一般都会使用**HashTable**或者**ConcurrentHashMap**，但是因为前者的**并发度**的原因基本上没啥使用场景了，所以存在线程不安全的场景我们都使用的是**ConcurrentHashMap**。
  
  HashTable直接在方法上锁，并发度很低，最多同时允许一个线程访问，所以用的不多了。
  
  
  
- 
## equals()和hashCode() 
- 为什么重写equals方法的时候需要重写hashCode方法？
  因为在java中，所有的对象都是继承于Object类。Object类中有两个方法equals、hashCode，这两个方法都是用来比较两个对象是否相等的。
  在未重写equals方法我们是继承了object的equals方法，那里的 equals是比较**两个对象的内存地址**，显然我们new了2个对象内存地址肯定不一样
  **对于值对象，==比较的是两个对象的值
  对于引用对象，比较的是两个对象的地址**
  哈希冲突的时候，拉链法将冲突的对象放在同一个链表里，查找到index后，需要使用equals方法去判断查找的是哪一个对象，所以如果我们对equals方法进行了重写，建议一定要对hashCode方法重写，以保证相同的对象返回相同的hash值，不同的对象返回不同的hash值。否则无法辨认。