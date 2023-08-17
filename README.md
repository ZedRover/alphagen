# Modified List
- alphagen_qlib
   1. stock_data_.py $\to$ load our data from shm
   2. calculator_.py $\to$ calculate IC and q9 
- alphagen/data/expression.py
   1. import from _.py
   2. Feature class*
- alphagen/models/model.py
   1. add ocean opt to *Operators List*


# TODO List
- [x] OPTOOL 重构-> alphagen/data/expression_ocean.py，继承下面四类运算符，实现对应的运算符(用 torch 替换 numpy)
  - UnaryOperator: 这种类型的操作符只接受一个操作数。例如，绝对值，符号，对数，CSRank（交叉部分排名）等。

  - BinaryOperator: 这种类型的操作符接受两个操作数。例如，加法，减法，乘法，除法，幂，较大，较小等。

  - RollingOperator: 这种类型的操作符接受一个操作数和一个时间差（delta time），在给定时间窗口内对操作数进行滚动操作。例如，平均，求和，标准差，方差，偏度等。

  - PairRollingOperator: 这种类型的操作符接受两个操作数和一个时间差，对这两个操作数在给定时间窗口内进行滚动操作。

- [x] New calculator, combine IC and q9
- [ ] Finish train process using new opts and features
- [ ] Speed optimization using cuda or cython

---------

# NOTE
- [alphagen/config.py](./alphagen/alphagen/config.py) 控制输入动作维度、最大长度等
- ...



