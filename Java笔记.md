# Java学习/复习笔记

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

## 继承extends
- 局部变量: 直接写成员变量名
  本类的成员变量: this.成员变量名
  父类的成员变量: super.成员变量名
- super关键字：用来访问父类内容
 1.访问父类的成员变量
 2.访问父类的成员方法
 3.访问父类的构造方法
- this关键字：用来访问本类内容
 1.访问本类的成员变量
 2.访问本类的另一个成员方法（可省略）
 3.访问本类的另一个构造方法（**注：必须是构造方法的第一句，且用了this就不再调用super**）
- Java是**单继承**的（即只能有一个直接父类），但可以**多级继承**
- Java里最高级的父类是java.lang.Object

## 重写（Overload）
- *重写（Overload）*：继承关系里，方法名称和参数列表都**一样**
  *重载（Overload）*：继承关系里，方法名称一样，参数列表**不一样**
- 在重写方法前加上*@Override*可以检测是否重写有效
- 子类方法的返回值必须**小于等于**父类方法的返回值范围（即可以返回父类返回类型的子类）
- 子类方法的权限必须**大于等于**父类方法的权限修饰符（**注：public>protected>(default)>private**）
- 父类方法里已有的代码行可以直接通过super.XX（）在子类重写方法里使用
- **super的父类构造调用必须写在子类构造方法的第一句且只能调用一次**

## 抽象（abstract）
- 抽象方法：带有abstract修饰符且不具有方法体的方法
- 抽象类：带有abstract修饰符的类
- 不能直接new抽象类对象，需要用子类来继承，再创建子类对象
- 抽象类可以有构造方法，是供子类创建对象时，初始化父类成员使用的
- 抽象类不一定包含抽象方法，但**抽象方法所在的类必须是抽象类**
- 子类必须覆盖重写（实现）抽象父类中**所有的**抽象方法，除非子类也是抽象类

## 接口（interface）
- 接口是多个类的公共规范
- 接口是一种引用数据类型，最重要的是其中的抽象方法
- 接口的格式 *public interface 接口名字{//接口内容}*
- 接口可以包含的内容：
-            1.Java7可以有：常量、抽象方法
-            2.Java8增加了默认方法、静态方法
-            3.Java9增加了私有方法
- 在任何版本中，接口都可以定义抽象方法*public abstract 返回值类型 方法名称*